"""Per-frame prep signals for one video -> data/features/<id>.parquet, <id>_tiny.npy, <id>_grid.npy"""
import sys, time
from pathlib import Path
import numpy as np
from rlairhockey.prep import frame_features

ROOT = Path(__file__).resolve().parents[1]
vid = sys.argv[1]
t0 = time.time()
df, tiny, grid = frame_features(ROOT / "data/proxy" / f"{vid}.mp4")
out = ROOT / "data/features"
df.to_parquet(out / f"{vid}.parquet")
np.save(out / f"{vid}_tiny.npy", tiny)
np.save(out / f"{vid}_grid.npy", grid)
print(vid, len(df), "frames", f"{time.time()-t0:.0f}s")
