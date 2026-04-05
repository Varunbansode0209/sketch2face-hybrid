from pathlib import Path

pair_files = [
    Path("data/processed/pairs/cufs.txt"),
    Path("data/processed/pairs/fs2k.txt"),
]

out = Path("data/processed/pairs/all_pairs.txt")
out.parent.mkdir(parents=True, exist_ok=True)

total = 0
with open(out, "w") as fout:
    for pf in pair_files:
        if not pf.exists():
            continue
        with open(pf) as fin:
            lines = fin.readlines()
            fout.writelines(lines)
            total += len(lines)

print("Total paired samples:", total)
