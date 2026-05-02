#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""HTTP client for the Kernel Arena REST API.

Usage:
    python arena_client.py catalog [--url URL]
    python arena_client.py pull --exemplar-id ID [--url URL]
    python arena_client.py submit --payload-file FILE [--url URL]
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

_DEFAULT_URL = "https://perf-bot.nvidia.com"
_TIMEOUT = 30  # seconds (arena uploads are not best-effort)


@functools.lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context that works with internal NVIDIA certs.

    Tries the default system CA bundle first.  If ``ARENA_SSL_VERIFY``
    is set to ``0`` or ``false``, or if no custom CA bundle is
    available, falls back to an unverified context so agents on
    internal networks can still reach the arena server.
    Lazily initialized on first use.
    """
    skip = os.environ.get("ARENA_SSL_VERIFY", "").lower() in ("0", "false")
    if skip:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    # Try default context first; if the server uses an internal CA the
    # handshake will fail, so we fall back to unverified.
    ctx = ssl.create_default_context()
    try:
        urllib.request.urlopen(
            urllib.request.Request(_DEFAULT_URL),
            timeout=5,
            context=ctx,
        )
        return ctx
    except (ssl.SSLError, urllib.error.URLError, OSError):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def _get(url: str) -> dict:
    """Make a GET request and return parsed JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_ssl_context()) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict) -> tuple[int, dict]:
    """Make a POST request and return (status_code, parsed_json)."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            req, timeout=_TIMEOUT, context=_ssl_context()
        ) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def cmd_catalog(args: argparse.Namespace) -> None:
    """Fetch and print the arena catalog."""
    url = f"{args.url}/api/v1/arena/catalog"
    try:
        data = _get(url)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_pull(args: argparse.Namespace) -> None:
    """Fetch and print an exemplar's kernel source."""
    url = f"{args.url}/api/v1/arena/exemplars/{args.exemplar_id}"
    try:
        data = _get(url)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit a kernel to the arena."""
    payload = json.loads(Path(args.payload_file).read_text())
    url = f"{args.url}/api/v1/arena/submit"
    status, data = _post(url, payload)
    print(json.dumps(data, indent=2))
    if status >= 400:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel Arena client")
    parser.add_argument(
        "--url",
        type=str,
        default=_DEFAULT_URL,
        help=f"Arena server URL (default: {_DEFAULT_URL})",
    )
    subs = parser.add_subparsers(dest="command")

    subs.add_parser("catalog", help="Fetch problem catalog")

    pull_p = subs.add_parser("pull", help="Pull an exemplar")
    pull_p.add_argument("--exemplar-id", required=True, help="Exemplar UUID")

    submit_p = subs.add_parser("submit", help="Submit a kernel")
    submit_p.add_argument(
        "--payload-file",
        required=True,
        help="Path to JSON payload file",
    )

    args = parser.parse_args()
    commands = {"catalog": cmd_catalog, "pull": cmd_pull, "submit": cmd_submit}
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands[args.command](args)


if __name__ == "__main__":
    main()
