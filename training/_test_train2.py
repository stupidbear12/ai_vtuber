"""Run finetune with captured output, long timeout, and periodic status."""
import subprocess, sys, os, time, threading

proc = subprocess.Popen(
    [sys.executable, "-u", "training/finetune_sion.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=r"C:\Users\thtgg\workspace2\ai_vtuber",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    text=True,
)

# Read output line by line and write to a capture file
capture_path = os.path.join(os.path.dirname(__file__), "finetune_capture.log")
with open(capture_path, "w", encoding="utf-8") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()

proc.wait()
# Append exit code
with open(capture_path, "a", encoding="utf-8") as f:
    f.write(f"\n=== EXIT CODE: {proc.returncode} ===\n")
