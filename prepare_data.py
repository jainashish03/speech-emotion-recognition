"""
Data transformation for SER: IEMOCAP wavs -> frame-level sequences for RNN / Transformer models.

Each utterance becomes a (T, 69) float16 matrix at a 10 ms hop:
  64 log-mel bands            spectral envelope
  log-F0 (speaker z-score)    pitch, 0 on unvoiced frames
  voiced flag                 1 on voiced frames
  RMS energy (dB)             loudness
  zero-crossing rate          noisiness / brightness
  spectral flux               frame-to-frame spectral change

Steps: 4-class label mapping -> load/resample 16 kHz -> trim silence -> crop at 10 s ->
features -> per-SPEAKER z-score (removes recording gain and speaker pitch range, keeps
within-speaker expressiveness; it uses no labels). Loudness is deliberately NOT normalised
per utterance because it is an emotion cue.

Usage:
  python prepare_data.py --csv iemocap_full_dataset.csv --root /path/to/IEMOCAP_full_release --out data
Outputs in --out: feats.npy (concatenated frames), offsets.npy, meta.csv, stats.json
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config  # all paths live in config.py (or override with --csv / --root / --out)

SR, HOP, NFFT, N_MELS = 16000, 160, 400, 64
FMIN, FMAX = 60.0, 450.0
MAX_SEC, MIN_FRAMES = 10, 50
LABELS = ["neutral", "happy", "sad", "angry"]
MAP = {"neu": "neutral", "hap": "happy", "exc": "happy", "sad": "sad", "ang": "angry"}
NAMES = [f"mel{i}" for i in range(N_MELS)] + ["logf0", "voiced", "rms_db", "zcr", "flux"]


def why(p):
    try:
        with open(p, "rb") as f:
            h = f.read(12)
        return "opens fine, but is not a valid WAV (header %r, %d bytes)" % (h[:4], os.path.getsize(p)) if h[:4] != b"RIFF" else "ok"
    except OSError as e:
        return f"{type(e).__name__}: {e} (mode {oct(os.stat(p).st_mode & 0o777) if os.path.exists(p) else 'n/a'})"


def unreadable(p):
    try:
        with open(p, "rb") as f:
            return f.read(4) != b"RIFF"
    except OSError:
        return True


def extract(path):
    """Raw (un-normalised) frame features: columns follow NAMES; logf0 is raw ln(Hz), 0 if unvoiced."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    y = y[: MAX_SEC * SR]
    if len(y) < MIN_FRAMES * HOP:
        y = np.pad(y, (0, MIN_FRAMES * HOP - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=8000)
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=None).clip(min=-100)  # (64, T)
    rms = librosa.feature.rms(y=y, frame_length=NFFT, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=NFFT, hop_length=HOP)[0]
    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=1024, hop_length=HOP)
    flux = np.maximum(np.diff(logmel, axis=1, prepend=logmel[:, :1]), 0).sum(0)

    T = min(logmel.shape[1], len(rms), len(zcr), len(f0))
    rms_db, zcr, f0, flux = rms_db[:T], zcr[:T], f0[:T], flux[:T]
    # voiced = loud enough relative to this utterance AND YIN not pinned at its search limits
    voiced = (rms_db > rms_db.max() - 30) & (f0 > FMIN * 1.02) & (f0 < FMAX * 0.98)
    logf0 = np.where(voiced, np.log(np.maximum(f0, 1.0)), 0.0)
    X = np.concatenate([logmel[:, :T].T, logf0[:, None], voiced[:, None], rms_db[:, None], zcr[:, None], flux[:, None]], 1)
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.CSV))
    ap.add_argument("--root", default=str(config.audio_root()), help="folder containing Session1..Session5")
    ap.add_argument("--out", default=str(config.DATA_DIR))
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N utterances")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    df = df[df.emotion.isin(MAP)].reset_index(drop=True)
    if a.limit:
        df = df.sample(a.limit, random_state=0).reset_index(drop=True)
    df["label_name"] = df.emotion.map(MAP)
    df["label"] = df.label_name.map(LABELS.index)
    if "speaker" not in df.columns:  # IEMOCAP: 10 actors = session + gender (1F, 1M, ..., 5M)
        df["speaker"] = df.session.astype(str) + df.gender
    df["speaker"] = df.speaker.astype(str)
    df["wav"] = [os.path.join(a.root, p) for p in df.path]
    missing = [p for p in df.wav if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} wav files not found, e.g. {missing[0]} -- check --root")
    bad = [p for p in df.wav if unreadable(p)]
    if bad:
        raise SystemExit(
            f"{len(bad)} of {len(df)} audio files cannot be read as WAV, e.g.\n  {bad[0]}\n  reason: {why(bad[0])}\n"
            f"Common fixes: permissions -> chmod -R u+rwX \"{a.root}\"  | incomplete download/unzip -> re-extract the archive  |\n"
            f"file still in iCloud/cloud storage -> download it fully (or move the folder out of the synced location).\n"
            + ("macOS 'Operation not permitted' with a normal mode (e.g. 0o664) is a macOS privacy/attribute block, not file permissions:\n"
               "  give your terminal app Full Disk Access (System Settings > Privacy & Security), restart it, or re-extract the\n"
               "  archive from Terminal into a folder outside Downloads (e.g. ~/datasets) and point AUDIO_ROOT there.\n"
               if sys.platform == "darwin" else ""))
    print(f"{len(df)} utterances; classes: {df.label_name.value_counts().to_dict()}")

    raw = Parallel(n_jobs=a.jobs, verbose=5)(delayed(extract)(p) for p in df.wav)
    raw = [r.astype(np.float16) for r in raw]  # halve RAM; stats still accumulated in float64

    # ---- per-speaker statistics (unsupervised) ----
    D = len(NAMES)
    i_f0, i_vo = N_MELS, N_MELS + 1
    stats = {}
    for spk, idx in df.groupby("speaker").indices.items():
        cat = np.concatenate([raw[i] for i in idx]).astype(np.float64)
        mu, sd = cat.mean(0), cat.std(0) + 1e-5
        v = cat[:, i_vo] > 0.5
        mu[i_f0], sd[i_f0] = (cat[v, i_f0].mean(), cat[v, i_f0].std() + 1e-5) if v.sum() > 10 else (0.0, 1.0)
        mu[i_vo], sd[i_vo] = 0.0, 1.0  # keep the 0/1 flag as is
        stats[spk] = {"mean": mu.tolist(), "std": sd.tolist()}

    # ---- normalise + pack into one array ----
    chunks, offsets = [], [0]
    for i, spk in enumerate(df.speaker):
        x = raw[i].astype(np.float32)
        mu, sd = np.array(stats[spk]["mean"], np.float32), np.array(stats[spk]["std"], np.float32)
        z = (x - mu) / sd
        voiced = x[:, i_vo] > 0.5
        z[:, i_f0] = np.where(voiced, z[:, i_f0], 0.0)  # unvoiced pitch = speaker mean = 0
        chunks.append(z.clip(-8, 8).astype(np.float16))
        offsets.append(offsets[-1] + len(z))
    feats = np.concatenate(chunks)
    offsets = np.array(offsets, dtype=np.int64)

    df.insert(0, "idx", np.arange(len(df)))
    df["n_frames"] = np.diff(offsets)
    np.save(f"{a.out}/feats.npy", feats)
    np.save(f"{a.out}/offsets.npy", offsets)
    df[["idx", "wav", "session", "speaker", "gender", "method", "label", "label_name", "agreement", "n_frames"]].to_csv(
        f"{a.out}/meta.csv", index=False)
    json.dump({"feature_names": NAMES, "speakers": stats, "labels": LABELS, "hop_ms": 10}, open(f"{a.out}/stats.json", "w"))
    print(f"saved {feats.shape} frames x {D} dims ({feats.nbytes / 1e6:.0f} MB), "
          f"mean length {df.n_frames.mean() / 100:.1f}s, max {df.n_frames.max() / 100:.1f}s")


if __name__ == "__main__":
    main()