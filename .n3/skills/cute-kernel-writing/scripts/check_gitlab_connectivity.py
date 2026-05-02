#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Check connectivity to NVIDIA internal GitLab for DKG repository access.

Used by ctm-kernel-writing and cute-kernel-writing skills which fetch content
from gitlab-master.nvidia.com via glab CLI.

Usage:
    python scripts/check_gitlab_connectivity.py --branch master
    python scripts/check_gitlab_connectivity.py          # all known branches
    python scripts/check_gitlab_connectivity.py --json         # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request

GITLAB_HOST = "gitlab-master.nvidia.com"
GITLAB_PROJECT = "dlarch-fastkernels%2Fdynamic-kernel-generator"
KNOWN_BRANCHES = {
    "master": "cute-kernel-writing, ctm-kernel-writing",
}
TREE_API = "projects/{project}/repository/tree?path=.&ref={branch}&per_page=1"


def check_https_reachable() -> dict:
    """Check if gitlab-master.nvidia.com is reachable over HTTPS."""
    url = f"https://{GITLAB_HOST}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10):
            return {"check": "https_reachable", "ok": True}
    except Exception as exc:
        return {"check": "https_reachable", "ok": False, "error": str(exc)}


def check_glab_installed() -> dict:
    """Check if glab CLI is installed and in PATH."""
    path = shutil.which("glab")
    if path:
        return {"check": "glab_installed", "ok": True, "path": path}
    return {
        "check": "glab_installed",
        "ok": False,
        "error": "glab not found in PATH. Install: https://gitlab.com/gitlab-org/cli",
    }


def check_glab_auth() -> dict:
    """Check if glab is authenticated to gitlab-master.nvidia.com."""
    try:
        result = subprocess.run(
            ["glab", "auth", "status", "--hostname", GITLAB_HOST],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {"check": "glab_auth", "ok": True}
        return {
            "check": "glab_auth",
            "ok": False,
            "error": (result.stderr or result.stdout).strip(),
        }
    except FileNotFoundError:
        return {"check": "glab_auth", "ok": False, "error": "glab not installed"}
    except subprocess.TimeoutExpired:
        return {"check": "glab_auth", "ok": False, "error": "timed out"}


def _branch_result(branch: str, *, ok: bool, error: str = "") -> dict:
    """Build a result dict for a branch-level check."""
    result = {
        "check": f"glab_api_{branch}",
        "ok": ok,
        "skill": KNOWN_BRANCHES.get(branch, "unknown"),
    }
    if not ok:
        result["error"] = error
    return result


def check_glab_api_branch(branch: str) -> dict:
    """Check if glab api can fetch from a specific branch."""
    endpoint = TREE_API.format(project=GITLAB_PROJECT, branch=branch)
    try:
        result = subprocess.run(
            ["glab", "api", "--hostname", GITLAB_HOST, endpoint],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, list) and data:
                return _branch_result(branch, ok=True)
        return _branch_result(
            branch, ok=False, error=(result.stderr or result.stdout).strip()[:200]
        )
    except FileNotFoundError:
        return _branch_result(branch, ok=False, error="glab not installed")
    except subprocess.TimeoutExpired:
        return _branch_result(branch, ok=False, error="timed out (>15s)")
    except json.JSONDecodeError:
        return _branch_result(branch, ok=False, error="invalid JSON response")


def run_checks(branches: list[str]) -> list[dict]:
    """Run all connectivity checks for the given branches."""
    results = [check_https_reachable()]

    glab_result = check_glab_installed()
    results.append(glab_result)
    if not glab_result["ok"]:
        for branch in branches:
            results.append(
                _branch_result(branch, ok=False, error="skipped (glab not installed)")
            )
        return results

    auth_result = check_glab_auth()
    results.append(auth_result)

    for branch in branches:
        if auth_result["ok"]:
            results.append(check_glab_api_branch(branch))
        else:
            results.append(
                _branch_result(branch, ok=False, error="skipped (not authenticated)")
            )

    return results


def print_human(results: list[dict]) -> None:
    """Print results in human-readable format."""
    print(f"GitLab connectivity: {GITLAB_HOST}")
    print(f"Project: {GITLAB_PROJECT.replace('%2F', '/')}")
    print()

    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        label = r["check"].replace("_", " ")
        skill = f" ({r['skill']})" if "skill" in r else ""
        print(f"  [{status}] {label}{skill}")
        if not r["ok"]:
            print(f"         {r.get('error', 'unknown error')}")

    print()
    if all(r["ok"] for r in results):
        print("All checks passed.")
    else:
        print("Some checks failed. Troubleshooting:")
        if not results[0]["ok"]:
            print(f"  - Verify network access to https://{GITLAB_HOST}")
            print("  - Check VPN connection if on corporate network")
        if len(results) > 1 and not results[1]["ok"]:
            print("  - Install glab: https://gitlab.com/gitlab-org/cli")
        if len(results) > 2 and not results[2]["ok"]:
            print(f"  - Run: glab auth login --hostname {GITLAB_HOST}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GitLab connectivity for DKG repo skills"
    )
    parser.add_argument(
        "--branch",
        help="Branch to check (default: all known branches)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    branches = [args.branch] if args.branch else list(KNOWN_BRANCHES)
    results = run_checks(branches)

    if args.json:
        ok = all(r["ok"] for r in results)
        print(json.dumps({"ok": ok, "checks": results}, indent=2))
    else:
        print_human(results)

    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
