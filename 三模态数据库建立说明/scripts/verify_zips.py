"""Verify all 120 BTS ZIP files are complete and valid"""
import os, zipfile

ZIP_DIR = r'd:\Daisy\bts_2016_2025'
expected = {(y, m) for y in range(2016, 2026) for m in range(1, 13)}

ok, empty, broken = 0, 0, 0
missing_years = {y: set() for y in range(2016, 2026)}

for y, m in sorted(expected):
    fname = f"{y}_{m:02d}.zip"
    path = os.path.join(ZIP_DIR, fname)
    if not os.path.isfile(path):
        missing_years[y].add(m)
        continue
    sz = os.path.getsize(path)
    if sz == 0:
        empty += 1
        missing_years[y].add((m, 'empty'))
        continue
    try:
        with zipfile.ZipFile(path) as z:
            csvs = [n for n in z.namelist() if n.endswith('.csv')]
            if not csvs:
                broken += 1
                print(f"  [WARN] {fname}: no CSV inside ({sz/1e6:.1f}MB)")
            else:
                ok += 1
    except Exception as e:
        broken += 1
        print(f"  [BROKEN] {fname}: {e}")

print(f"\nTotal: {ok+empty+broken} | OK: {ok} | Empty: {empty} | Broken: {broken} | Missing: {120-ok-empty-broken}")

# 输出每年摘要
for y in range(2016, 2026):
    miss = missing_years[y]
    if miss:
        print(f"  {y}: missing months {sorted(miss)}")
    else:
        print(f"  {y}: all 12 OK")
