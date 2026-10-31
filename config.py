r"""
THE ONLY FILE YOU NEED TO EDIT.
Paths are relative to this folder unless you write a full path, e.g. r"D:\datasets\IEMOCAP_full_release".
"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ============================== EDIT THESE ==============================
DATASET = "iemocap"            # "iemocap" | "cremad" | "ravdess"
AUDIO_ROOT = BASE / "/Users/ashish/Downloads/IEMOCAP_full_release"    # where your audio lives. Default: the "audio" folder next to this file.
                               # IEMOCAP: the folder holding Session1..Session5 (found automatically
                               #          even if it is a few folders deeper than AUDIO_ROOT).
                               # CREMA-D / RAVDESS: any folder containing the .wav files.
MODELS = ["bilstm", "transformer"]   # add "wavlm" once your GPU is ready
# ========================================================================

CSV = BASE / ("iemocap_full_dataset.csv" if DATASET == "iemocap" else f"labels_{DATASET}.csv")
DATA_DIR = BASE / f"data_{DATASET}"     # prepared features go here
RUNS_DIR = BASE / f"runs_{DATASET}"     # trained models + results go here
WORKERS = 0 if os.name == "nt" else 2   # data-loader processes (0 is safest on Windows)


def audio_root():
    root = Path(AUDIO_ROOT)
    if DATASET != "iemocap":
        return root
    cands = [root] + [p for p in root.glob("*") if p.is_dir()] + [p for p in root.glob("*/*") if p.is_dir()]
    for c in cands:
        if (c / "Session1").is_dir():
            return c
    return root
