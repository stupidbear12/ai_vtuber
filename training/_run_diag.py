"""Run diagnostic training and capture all output."""
import subprocess, sys, os

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_capture.log")

proc = subprocess.Popen(
    [sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_train.py")],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=r"C:\Users\thtgg\workspace2\ai_vtuber",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

with open(log_path, "w", encoding="utf-8") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()

proc.wait()
with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"\n=== EXIT CODE: {proc.returncode} ===\n")
