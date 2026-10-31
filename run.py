"""
One entry point.  Run from this folder:
  python run.py            check your setup (start here)
  python run.py prepare    labels CSV + feature extraction
  python run.py baseline   quick SVM on handcrafted DSP features
  python run.py train      train every model listed in config.MODELS
  python run.py all        check -> prepare -> train
  python run.py dashboard  open the web dashboard
Smoke test first:  python run.py all --limit 200 --epochs 2 --folds 1
"""
import argparse
import importlib
import json
import subprocess
import sys
import webbrowser

import config

PY, B = sys.executable, config.BASE


def run(script, *args):
    cmd = [PY, str(B / script), *map(str, args)]
    print("\n>", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=B).returncode:
        sys.exit(f"\nStep failed: {script}. Read the error above it.")


def have(m):
    try:
        importlib.import_module(m)
        return True
    except Exception:
        return False


def check():
    ok_data = ok_train = True
    print(f"Python {sys.version.split()[0]} | dataset: {config.DATASET} | folder: {B}")
    need = ["numpy", "pandas", "sklearn", "scipy", "joblib", "librosa", "soundfile"]
    miss = [m for m in need if not have(m)]
    print(("  [ok] " if not miss else "  [!!] missing: " + ", ".join(miss) + " -> ") +
          ("data packages installed" if not miss else "run: pip install -r requirements.txt"))
    ok_data &= not miss
    if have("torch"):
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (training will use the CPU, slowly)"
        print(f"  [ok] torch {torch.__version__} | GPU: {gpu}")
    else:
        print("  [!!] torch not installed (needed for training) -> pip install -r requirements.txt")
        ok_train = False
    if "wavlm" in config.MODELS and not have("transformers"):
        print("  [!!] 'wavlm' is in MODELS but `transformers` is missing -> pip install transformers")
        ok_train = False

    root = config.audio_root()
    if config.DATASET == "iemocap":
        if not config.CSV.exists():
            print(f"  [!!] labels CSV not found: {config.CSV}")
            return False, False
        import pandas as pd
        paths = pd.read_csv(config.CSV, usecols=["path"]).path
        s = paths.sample(min(300, len(paths)), random_state=0)
        n = sum((root / p).exists() for p in s)
        what = f"{n}/{len(s)} sampled IEMOCAP audio files found under {root}"
    else:
        n = sum(1 for _ in root.rglob("*.wav")) if root.exists() else 0
        what = f"{n} .wav files found under {root}"
    print(f"  [{'ok' if n else '!!'}] {what}")
    if not n:
        print("       Put the audio inside the 'audio' folder (see audio/PUT_AUDIO_HERE.txt) or set AUDIO_ROOT in config.py")
    ok_data &= bool(n)
    print("\nReady to prepare." if ok_data else "\nFix the [!!] items above, then run `python run.py` again.")
    return ok_data, ok_data and ok_train


def dirs(a):
    s = "_smoke" if a.limit else ""
    return str(config.DATA_DIR) + s, str(config.RUNS_DIR) + s


def prepare(a):
    if config.DATASET != "iemocap":
        run("make_csv.py")
    run("prepare_data.py", "--out", dirs(a)[0], *(["--limit", a.limit] if a.limit else []))


def train(a):
    data, runs = dirs(a)
    for m in config.MODELS:
        extra = (["--epochs", a.epochs] if a.epochs else []) + (["--folds", *a.folds] if a.folds else [])
        run("train.py", "--model", m, "--data", data, "--out", runs, *extra)
    print("\n=== SUMMARY (leave-one-fold-out) ===")
    for m in config.MODELS:
        try:
            r = json.load(open(f"{runs}/{m}/results.json"))["overall"]
            print(f"  {m:12s} UA {r['ua']:.3f}  WA {r['wa']:.3f}  F1 {r['f1']:.3f}")
        except Exception:
            print(f"  {m:12s} no results")
    print(f"Details: {runs}/<model>/results.json  (and dashboard_results.json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="check", choices=["check", "prepare", "baseline", "train", "all", "dashboard"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N utterances (separate *_smoke folders)")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--folds", nargs="+", default=None)
    a = ap.parse_args()

    if a.cmd == "check":
        check()
    elif a.cmd == "dashboard":
        run("server.py")
    elif a.cmd == "baseline":
        run("ser_pipeline.py", *(["--limit", a.limit] if a.limit else []))
    elif a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "train":
        train(a)
    else:
        ok_data, ok_train = check()
        if not ok_train:
            sys.exit("\nStopped: fix the [!!] items first.")
        prepare(a)
        train(a)


if __name__ == "__main__":
    main()
