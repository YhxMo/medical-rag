"""Read credentials without echo; retain them only in memory for bounded repair retries."""

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.application.client import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Continue in a local background process after secure entry",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--retry-window",
        type=int,
        default=600,
        help="Memory-only retry window after failure, seconds",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Python script or -m module and arguments",
    )
    args = parser.parse_args()
    if not args.command:
        parser.error("Supply a Python script or -m module command")
    if not 0 <= args.retry_window <= 1800:
        parser.error("Retry window must be 0..1800 seconds")
    if not args.worker and not sys.stdin.isatty():
        parser.error("An interactive local terminal is required")
    env = os.environ.copy()
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        if not env.get(name):
            if args.worker:
                parser.error("Background worker requires in-memory credentials")
            env[name] = getpass.getpass(f"{name} (hidden; Enter to skip): ").strip()
    base = ROOT / "artifacts/resume-v2"
    state_path = base / "secure_runner_state.json"
    signal = base / "retry.signal"
    attempt = 0
    try:
        if args.detach and not args.worker:
            base.mkdir(parents=True, exist_ok=True)
            env["PYTHONUNBUFFERED"] = "1"
            with (base / "workflow.log").open("a") as log:
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--retry-window",
                        str(args.retry_window),
                        *args.command,
                    ],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(
                f"Local workflow started (PID {worker.pid}); progress: {base / 'workflow.log'}"
            )
            return 0
        while True:
            attempt += 1
            atomic_json(
                state_path,
                {"pid": os.getpid(), "status": "running", "attempt": attempt},
            )
            code = subprocess.call([sys.executable, *args.command], cwd=ROOT, env=env)
            if code == 0 or not args.retry_window:
                atomic_json(
                    state_path,
                    {
                        "pid": os.getpid(),
                        "status": "completed" if code == 0 else "failed",
                        "exit_code": code,
                    },
                )
                return code
            deadline = time.monotonic() + args.retry_window
            previous = signal.stat().st_mtime_ns if signal.exists() else 0
            atomic_json(
                state_path,
                {
                    "pid": os.getpid(),
                    "status": "awaiting_repair",
                    "attempt": attempt,
                    "exit_code": code,
                    "expires_in_seconds": args.retry_window,
                },
            )
            print(
                f"Workflow paused. Credentials remain in this process memory for {args.retry_window}s. "
                "The assistant can repair code and request a retry. Ctrl+C exits and clears credentials.",
                flush=True,
            )
            while time.monotonic() < deadline:
                if signal.exists() and signal.stat().st_mtime_ns > previous:
                    break
                time.sleep(1)
            else:
                atomic_json(
                    state_path,
                    {"pid": os.getpid(), "status": "expired", "exit_code": code},
                )
                return code
    finally:
        for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            env.pop(name, None)


if __name__ == "__main__":
    raise SystemExit(main())
