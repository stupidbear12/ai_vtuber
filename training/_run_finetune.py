"""Wrapper: runs finetune_sion.py via subprocess and captures all output."""
import subprocess, sys, os

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(TRAINING_DIR, "finetune_sion.py")
capture_log = os.path.join(TRAINING_DIR, "finetune_capture.log")

proc = subprocess.Popen(
    [sys.executable, "-u", script],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=r"C:\Users\thtgg\workspace2\ai_vtuber",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

with open(capture_log, "w", encoding="utf-8") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()

proc.wait()
with open(capture_log, "a", encoding="utf-8") as f:
    f.write(f"\n=== EXIT CODE: {proc.returncode} ===\n")
