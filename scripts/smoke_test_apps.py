"""Smoke test script for AquaVision Streamlit applications.

Starts each app in headless mode on a unique port, queries its health check
endpoint to ensure it boots without errors, and cleanly terminates the process.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request


APPS = [
    ("app.py", 8511),
    ("streamlit_app.py", 8512),
    ("biomass_app.py", 8513),
]


def test_app(script_name: str, port: int) -> bool:
    print(f"\n--- Testing {script_name} on port {port} ---")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script_name,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    start_time = time.time()
    success = False

    try:
        while time.time() - start_time < 30:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print(f"FAILED: Process exited prematurely with code {proc.returncode}")
                print(f"Stdout:\n{stdout[:500]}")
                print(f"Stderr:\n{stderr[:500]}")
                return False
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"SUCCESS: {script_name} responded HTTP 200 on {health_url}")
                        success = True
                        break
            except Exception:
                time.sleep(1)
        if not success:
            print(f"TIMEOUT: {script_name} did not respond within 30 seconds.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return success


def main() -> int:
    all_ok = True
    for script_name, port in APPS:
        ok = test_app(script_name, port)
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll 3 Streamlit applications passed smoke test successfully!")
        return 0
    else:
        print("\nOne or more Streamlit applications failed smoke test.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
