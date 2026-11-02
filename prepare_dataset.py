import shutil
from pathlib import Path
import pandas as pd

OUT = Path("output")
PROC = OUT / "resized_160"
DEST = Path("dataset")

split = pd.read_csv(OUT / "split.csv")                    # path, label, split
meta = pd.read_csv(OUT / "meta.csv")[["path", "md5"]]     # untuk nama file hasil resize
df = split.merge(meta, on="path", how="left")

ok = miss = 0
for r in df.itertuples():
    src = PROC / r.label / f"{Path(r.path).stem}_{str(r.md5)[:8]}.jpg"
    dst = DEST / r.split / r.label / src.name
    if not src.exists():
        miss += 1
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    ok += 1

print(f"Tersalin: {ok} | tidak ditemukan: {miss}")
print(df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int))