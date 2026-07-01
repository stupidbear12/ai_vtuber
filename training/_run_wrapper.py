import subprocess, sys
result = subprocess.run(
    [sys.executable, "-u", "training/finetune_sion.py"],
    capture_output=True, text=True, cwd=r"C:\Users\thtgg\workspace2\ai_vtuber"
)
print("STDOUT:", result.stdout[:3000] if result.stdout else "(empty)")
print("STDERR:", result.stderr[:3000] if result.stderr else "(empty)")
print("RETURN CODE:", result.returncode)
