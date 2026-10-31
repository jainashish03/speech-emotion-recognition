"""
Speech Emotion Recognition on IEMOCAP: prosodic/spectral features + leave-one-session-out CV.

Usage:
  python run.py baseline        (paths come from config.py)
  python ser_pipeline.py --features wavlm   # SSL embeddings (needs torch + transformers)

--root is the folder that contains Session1/ ... Session5/ (the CSV `path` column is relative to it).
Protocol: 4 classes (neutral, happy+excited, sad, angry), 5531 utterances, 5-fold leave-one-session-out.
Metrics: UA (unweighted accuracy = macro recall), WA (accuracy), macro-F1.
"""
import argparse
import os

import numpy as np
import pandas as pd

import config
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SR = 16000
LABELS = ["neutral", "happy", "sad", "angry"]
MAP = {"neu": "neutral", "hap": "happy", "exc": "happy", "sad": "sad", "ang": "angry"}


def load_meta(csv):
    df = pd.read_csv(csv)
    df = df[df.emotion.isin(MAP)].copy()
    df["y"] = df.emotion.map(MAP).map(LABELS.index)
    return df.reset_index(drop=True)


# ---------- handcrafted features (DSP) ----------
def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return [x.mean(), x.std(), x.min(), x.max(), np.percentile(x, 25), np.percentile(x, 75)]


def handcrafted(path):
    import librosa

    y, _ = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    f = []

    # Spectral envelope: MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
    d1 = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] | 1))
    for m in (mfcc, d1):
        f += list(m.mean(1)) + list(m.std(1))

    # Prosody: pitch (YIN), voiced ratio, energy, rhythm proxy
    f0 = librosa.yin(y, fmin=60, fmax=450, sr=SR, frame_length=1024)
    HOP = 256  # same hop as YIN so pitch and energy frames line up
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    m = min(len(f0), len(rms))
    voiced = rms[:m] > 0.1 * rms.max()
    f0v = f0[:m][voiced] if voiced.any() else f0
    f += _stats(f0v) + [np.ptp(f0v)]
    f += _stats(np.diff(f0v) if len(f0v) > 1 else [0.0])
    f += _stats(rms) + _stats(librosa.amplitude_to_db(rms + 1e-8))
    f += [voiced.mean(), len(y) / SR]

    # Prosody: pauses, speech-segment rate, intonation slope (Hz/s), pitch jitter
    dur = len(y) / SR
    segs = int(np.sum(np.diff(voiced.astype(int)) == 1)) + int(voiced[0])
    t = np.arange(len(f0v)) * HOP / SR
    slope = np.polyfit(t, f0v, 1)[0] if len(f0v) > 2 else 0.0
    jitter = np.mean(np.abs(np.diff(f0v))) / (f0v.mean() + 1e-8) if len(f0v) > 1 else 0.0
    f += [1.0 - voiced.mean(), segs / dur, slope, jitter]

    # Spectral shape
    for feat in (
        librosa.feature.zero_crossing_rate(y)[0],
        librosa.feature.spectral_centroid(y=y, sr=SR)[0],
        librosa.feature.spectral_bandwidth(y=y, sr=SR)[0],
        librosa.feature.spectral_rolloff(y=y, sr=SR)[0],
    ):
        f += [feat.mean(), feat.std()]
    f += list(librosa.feature.spectral_contrast(y=y, sr=SR).mean(1))
    f += _stats(librosa.onset.onset_strength(y=y, sr=SR))  # spectral flux
    return np.nan_to_num(np.array(f, dtype=np.float32))


# ---------- SSL embeddings (optional; untested without GPU/model download) ----------
def wavlm_embeddings(paths, model_name="microsoft/wavlm-base-plus", batch_log=200):
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            y, _ = librosa.load(p, sr=SR)
            inp = fe(y, sampling_rate=SR, return_tensors="pt").to(dev)
            h = model(**inp, output_hidden_states=True).hidden_states
            # mean over time, then average a few mid/upper layers (emotion info sits mid-stack)
            e = torch.stack([h[k].mean(1).squeeze(0) for k in (6, 8, 10, 12)]).mean(0)
            out.append(e.cpu().numpy())
            if i % batch_log == 0:
                print(f"  wavlm {i}/{len(paths)}", flush=True)
    return np.stack(out)


def build_features(df, root, kind, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        if z["X"].shape[0] == len(df):
            print(f"loaded cached features {z['X'].shape} from {cache}")
            return z["X"]
    paths = [os.path.join(root, p) for p in df.path]
    if kind == "handcrafted":
        X = np.stack(Parallel(n_jobs=-1, verbose=5)(delayed(handcrafted)(p) for p in paths))
    else:
        X = wavlm_embeddings(paths)
    np.savez_compressed(cache, X=X)
    return X


def loso_eval(X, df):
    y = df.y.values
    sess = df.session.values
    pred = np.zeros_like(y)
    rows = []
    for s in sorted(np.unique(sess)):
        te, tr = sess == s, sess != s
        clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        rows.append((s, recall_score(y[te], pred[te], average="macro"), accuracy_score(y[te], pred[te])))
        print(f"  held-out Session{s}: UA={rows[-1][1]:.3f}  WA={rows[-1][2]:.3f}")
    print(f"\nOverall  UA={recall_score(y, pred, average='macro'):.3f}  "
          f"WA={accuracy_score(y, pred):.3f}  macroF1={f1_score(y, pred, average='macro'):.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):", LABELS)
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.CSV))
    ap.add_argument("--root", default=str(config.audio_root()))
    ap.add_argument("--features", choices=["handcrafted", "wavlm"], default="handcrafted")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N rows")
    a = ap.parse_args()

    df = load_meta(a.csv)
    if a.limit:
        df = df.sample(a.limit, random_state=0).reset_index(drop=True)
    print(f"{len(df)} utterances, class counts: {df.y.map(dict(enumerate(LABELS))).value_counts().to_dict()}")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    X = build_features(df, a.root, a.features, str(config.DATA_DIR / f"baseline_{a.features}_{len(df)}.npz"))
    print(f"feature matrix: {X.shape}")
    loso_eval(X, df)
