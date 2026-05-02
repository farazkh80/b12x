#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Verify kernel correctness against a reference implementation.

Standalone script -- only Python stdlib required (torch needed at runtime for
GPU verification, but not for --mock mode).
Outputs structured JSON to stdout.

Usage:
    python verify_kernel.py --kernel-path kernel.py \\
        --reference-code "def ref(x): return x * 2" \\
        --input-shapes '{"x": [1024]}' \\
        --input-dtypes '{"x": "float32"}' \\
        [--rtol 1e-3] [--atol 1e-3] \\
        [--env-vars '{"ENABLE_TILE": "1"}'] \\
        [--timeout 60] \\
        [--mock]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------

_INTEGER_DTYPES = {"int8", "int16", "int32", "int64", "uint8"}
_FLOAT8_DTYPES = {"float8_e4m3fn", "float8_e5m2"}
_DTYPE_MAP = {
    name: f"torch.{name}"
    for name in [
        "float16",
        "float32",
        "bfloat16",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "float8_e4m3fn",
        "float8_e5m2",
    ]
}


def _build_input_creation_code(
    input_shapes: dict[str, list[int]],
    input_dtypes: dict[str, str],
) -> str:
    """Build Python code for creating test input tensors."""
    lines: list[str] = []
    for name, shape in input_shapes.items():
        dtype = input_dtypes.get(name, "float32")
        torch_dtype = _DTYPE_MAP.get(dtype, "torch.float32")
        if dtype in _INTEGER_DTYPES:
            lines.append(
                f'inputs["{name}"] = torch.randint('
                f'-10, 10, {shape}, dtype={torch_dtype}, device="cuda")'
            )
        elif dtype in _FLOAT8_DTYPES:
            lines.append(
                f'inputs["{name}"] = torch.randn('
                f'{shape}, dtype=torch.float16, device="cuda").to({torch_dtype})'
            )
        else:
            lines.append(
                f'inputs["{name}"] = torch.randn('
                f'{shape}, dtype={torch_dtype}, device="cuda")'
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reference code sanitization
# ---------------------------------------------------------------------------


def _sanitize_reference_code(reference_code: str) -> tuple[str, str]:
    """Sanitize reference code for safe insertion into verification scripts.

    Expands F.xxx -> torch.nn.functional.xxx and extracts indented imports
    to module level.

    Returns:
        Tuple of (extra_imports_str, cleaned_reference_code).
    """
    # Expand F.xxx -> torch.nn.functional.xxx
    cleaned_code = re.sub(
        r"\bF\.(\w+)",
        r"torch.nn.functional.\1",
        reference_code,
    )

    # Strip `import torch.nn.functional as F` since F is no longer used
    cleaned_code = re.sub(
        r"^\s*import torch\.nn\.functional as F\s*$",
        "",
        cleaned_code,
        flags=re.MULTILINE,
    )

    # Extract indented imports to module level
    lines = cleaned_code.split("\n")
    import_lines: list[str] = []
    code_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and line != line.lstrip():
            import_lines.append(stripped)
        else:
            code_lines.append(line)

    extra_imports = "\n".join(import_lines)
    cleaned_code = "\n".join(code_lines)

    return extra_imports, cleaned_code


# ---------------------------------------------------------------------------
# Verification harness generation
# ---------------------------------------------------------------------------


def _build_verification_script(
    kernel_path: str,
    reference_code: str,
    input_shapes: dict[str, list[int]],
    input_dtypes: dict[str, str],
    rtol: float,
    atol: float,
) -> str:
    """Generate a temporary verification harness script.

    The harness:
    1. Imports the kernel module
    2. Defines the reference function
    3. Creates inputs from shapes/dtypes
    4. Runs both and compares outputs
    """
    extra_imports, cleaned_reference_code = _sanitize_reference_code(reference_code)
    input_code = _build_input_creation_code(input_shapes, input_dtypes)

    script = f"""\
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "{os.path.dirname(os.path.abspath(kernel_path))}")

# Import the generated kernel module
import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "{kernel_path}")
kernel_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_module)

# Re-import after kernel module (guards against namespace clobbering)
import torch.nn.functional as F  # noqa: F811
{extra_imports}

# Snapshot globals before reference code (for ref_fn discovery)
_ref_globals_snapshot = set(globals().keys())

# Define reference implementation
{cleaned_reference_code}

# Create test inputs
torch.manual_seed(42)
inputs = {{}}
{input_code}

# Verify TileIR backend if ENABLE_TILE=1 is set
import os
_enable_tile = os.environ.get('ENABLE_TILE', '0')
_actual_backend = 'unknown'
try:
    import triton.runtime.driver
    _target = triton.runtime.driver.active.get_current_target()
    _actual_backend = getattr(_target, 'backend', 'unknown')
    print(f"BACKEND_INFO:enable_tile={{_enable_tile}},actual={{_actual_backend}}")
except Exception:
    print(f"BACKEND_INFO:enable_tile={{_enable_tile}},actual=unknown")

# Discover wrapper function from kernel module
wrapper_fn = None

for preferred_name in ["run_kernel", "forward"]:
    if hasattr(kernel_module, preferred_name):
        obj = getattr(kernel_module, preferred_name)
        if callable(obj):
            wrapper_fn = obj
            break

if wrapper_fn is None:
    for name in dir(kernel_module):
        if name.startswith("fused_") and callable(getattr(kernel_module, name)):
            wrapper_fn = getattr(kernel_module, name)
            break

if wrapper_fn is None:
    excluded_names = {{"torch", "triton", "tl", "cute", "cutlass", "importlib", "sys"}}
    for name in dir(kernel_module):
        obj = getattr(kernel_module, name)
        if not callable(obj) or name.startswith("_"):
            continue
        if name in excluded_names:
            continue
        if name.endswith("_kernel") or name.endswith("_host"):
            continue
        wrapper_fn = obj
        break

if wrapper_fn is None:
    print("ERROR:No wrapper function found in kernel module")
    sys.exit(1)

# Discover reference function
import types as _types
_discovered_ref_fn = None
_new_global_names = set(globals().keys()) - _ref_globals_snapshot
for _name in sorted(_new_global_names):
    if _name.startswith("_") or _name in ("wrapper_fn", "inputs"):
        continue
    _obj = globals().get(_name)
    if not isinstance(_obj, _types.FunctionType):
        continue
    if hasattr(kernel_module, _name):
        continue
    _discovered_ref_fn = _obj
    break

if _discovered_ref_fn is None:
    for _name, _obj in list(globals().items()):
        if not isinstance(_obj, _types.FunctionType):
            continue
        if _name.startswith("_"):
            continue
        if _name in ("wrapper_fn",):
            continue
        if hasattr(kernel_module, _name):
            continue
        _discovered_ref_fn = _obj
        break

if _discovered_ref_fn is None:
    print("ERROR:No reference function found in globals")
    sys.exit(1)

# Run both implementations and compare
try:
    input_list = list(inputs.values())
    kernel_out = wrapper_fn(*input_list)
    _ref_result = _discovered_ref_fn(*input_list)

    if isinstance(kernel_out, tuple):
        kernel_out = kernel_out[0]
    if isinstance(_ref_result, tuple):
        _ref_result = _ref_result[0]

    abs_diff = (kernel_out.float() - _ref_result.float()).abs()
    max_abs = abs_diff.max().item()

    # Compute relative diff avoiding division by zero
    ref_abs = _ref_result.float().abs()
    safe_ref = torch.where(ref_abs > 0, ref_abs, torch.ones_like(ref_abs))
    max_rel = (abs_diff / safe_ref).max().item()

    _kf = kernel_out.float()
    _rf = _ref_result.float()
    passed = torch.allclose(_kf, _rf, rtol={rtol}, atol={atol})

    print(f"RESULT:passed={{passed}},max_abs={{max_abs}},max_rel={{max_rel}}")
except Exception as e:
    print(f"ERROR:{{e}}")
    sys.exit(1)
"""
    return script


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------


def verify_kernel(
    kernel_path: str,
    reference_code: str,
    input_shapes: dict[str, list[int]],
    input_dtypes: dict[str, str],
    rtol: float = 1e-3,
    atol: float = 1e-3,
    env_vars: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict:
    """Verify kernel correctness against a reference implementation.

    Args:
        kernel_path: Path to kernel Python file.
        reference_code: Python code defining the reference function.
        input_shapes: Map of input name to shape list.
        input_dtypes: Map of input name to dtype string.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        env_vars: Additional environment variables.
        timeout: Execution timeout in seconds.

    Returns:
        Verification result dict matching the output schema.
    """
    if not os.path.exists(kernel_path):
        print(f"Kernel file not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    script = _build_verification_script(
        kernel_path, reference_code, input_shapes, input_dtypes, rtol, atol
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(script)
        script_path = script_file.name

    working_dir = os.path.dirname(os.path.abspath(kernel_path)) or "."

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=env,
        )

        output = result.stdout + result.stderr

        # Parse backend info
        backend_info = ""
        for line in output.split("\n"):
            if "BACKEND_INFO:" in line:
                backend_info = line.split("BACKEND_INFO:")[1].strip()

        if "RESULT:" in output:
            result_line = [line for line in output.split("\n") if "RESULT:" in line][0]
            parts_str = result_line.split("RESULT:")[1]
            parts = parts_str.split(",")
            passed = "True" in parts[0]
            max_abs = float(parts[1].split("=")[1])
            max_rel = float(parts[2].split("=")[1])

            if passed:
                details = (
                    f"All outputs match within tolerance " f"(rtol={rtol}, atol={atol})"
                )
            else:
                details = (
                    f"Outputs differ beyond tolerance "
                    f"(max_abs={max_abs:.2e}, rtol={rtol}, atol={atol})"
                )

            return {
                "correct": passed,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "backend_info": backend_info,
                "details": details,
            }
        elif "ERROR:" in output:
            error_msg = output.split("ERROR:")[1].strip().split("\n")[0]
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "backend_info": backend_info,
                "details": f"Verification error: {error_msg}",
            }
        else:
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "backend_info": backend_info,
                "details": f"Unexpected output: {output[:500]}",
            }

    except subprocess.TimeoutExpired:
        return {
            "correct": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "backend_info": "",
            "details": f"Verification timed out after {timeout} seconds",
        }
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data(rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    """Return realistic mock verification data for testing."""
    return {
        "correct": True,
        "max_abs_diff": 1.2e-7,
        "max_rel_diff": 3.4e-6,
        "backend_info": "enable_tile=0,actual=cuda",
        "details": (f"All outputs match within tolerance (rtol={rtol}, atol={atol})"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify kernel correctness against a reference."
    )
    parser.add_argument(
        "--kernel-path",
        help="Path to Python file containing the kernel.",
    )
    parser.add_argument(
        "--reference-code",
        help="Python code defining the reference function.",
    )
    parser.add_argument(
        "--input-shapes",
        help="JSON dict mapping input names to shape lists.",
    )
    parser.add_argument(
        "--input-dtypes",
        help="JSON dict mapping input names to dtype strings.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--env-vars",
        help="JSON dict of environment variables.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Execution timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing.",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data(rtol=args.rtol, atol=args.atol)
    elif args.kernel_path:
        # Parse JSON inputs
        try:
            input_shapes = json.loads(args.input_shapes)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Invalid JSON for --input-shapes: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            input_dtypes = json.loads(args.input_dtypes)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Invalid JSON for --input-dtypes: {e}", file=sys.stderr)
            sys.exit(1)

        env_vars = None
        if args.env_vars:
            try:
                env_vars = json.loads(args.env_vars)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON for --env-vars: {e}", file=sys.stderr)
                sys.exit(1)

        data = verify_kernel(
            kernel_path=args.kernel_path,
            reference_code=args.reference_code or "",
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            rtol=args.rtol,
            atol=args.atol,
            env_vars=env_vars,
            timeout=args.timeout,
        )
    else:
        parser.error("Either --mock or --kernel-path is required.")

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
