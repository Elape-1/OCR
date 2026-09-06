#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
        return False


def find_pid_for_port(port: int) -> str:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique) 2>$null",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
        return ", ".join(sorted(set(pids))) if pids else "unknown"

    if os.name == "posix":
        for command in (
            ["lsof", "-nP", "-iTCP:" + str(port), "-sTCP:LISTEN"],
            ["ss", "-ltnp", "sport = :" + str(port)],
        ):
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[0]

    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a local port is available before starting the backend.")
    parser.add_argument("--port", type=int, default=8000, help="Port to check (default: 8000)")
    args = parser.parse_args()

    if port_in_use(args.port):
        pid_text = find_pid_for_port(args.port)
        print(
            f"Port {args.port} already in use by PID {pid_text} — kill it first or choose another port.",
            file=sys.stderr,
        )
        return 1

    print(f"Port {args.port} is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
