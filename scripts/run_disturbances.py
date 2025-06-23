from diafiltration.disturbance import run_mismatch
for f in (0.75, 0.5, 0.25):
    hrs = run_mismatch(f)
    print(f"true kM_L = {f:4.2f} → batch {hrs:5.2f} h")
