"""
Build a labels CSV (same columns as iemocap_full_dataset.csv) from a free audio dataset,
so prepare_data.py and train.py run unchanged.

  python make_csv.py --dataset cremad  --root "D:\\datasets\\CREMA-D"  --out cremad.csv
  python make_csv.py --dataset ravdess --root "D:\\datasets\\RAVDESS"  --out ravdess.csv
  python prepare_data.py --csv cremad.csv --root "D:\\datasets\\CREMA-D" --out data_cremad
  python train.py --model bilstm --data data_cremad

--root is any folder that contains the .wav files (sub-folders are searched).
Labels are mapped to the same 4 classes as IEMOCAP (neutral, happy, sad, angry); other emotions are dropped.
The 'session' column holds a speaker-independent fold id 1..5 (no actor appears in two folds), so the
leave-one-fold-out evaluation in train.py is speaker-independent. 'speaker' is the actor id.

CREMA-D file:  1001_DFA_ANG_XX.wav          actor_sentence_emotion_intensity   (91 actors, ~4,900 clips in 4 classes)
RAVDESS file:  03-01-05-01-02-01-12.wav     modality-channel-emotion-intensity-statement-repetition-actor
               (speech only; 'calm' dropped; odd actor id = male)  (~670 clips in 4 classes, 24 actors)
"""
import argparse
import os

import pandas as pd

import config

CREMAD = {"NEU": "neu", "HAP": "hap", "SAD": "sad", "ANG": "ang"}
RAVDESS = {"01": "neu", "03": "hap", "04": "sad", "05": "ang"}


def scan(root):
    for d, _, fs in os.walk(root):
        for f in sorted(fs):
            if f.lower().endswith(".wav"):
                yield os.path.join(d, f)


def rows_cremad(root):
    for p in scan(root):
        parts = os.path.splitext(os.path.basename(p))[0].split("_")
        if len(parts) >= 3 and parts[2] in CREMAD:
            yield dict(path=p, speaker=parts[0], emotion=CREMAD[parts[2]], gender="U", method="acted")


def rows_ravdess(root):
    for p in scan(root):
        parts = os.path.splitext(os.path.basename(p))[0].split("-")
        if len(parts) == 7 and parts[1] == "01" and parts[2] in RAVDESS:  # vocal channel 01 = speech
            actor = int(parts[6])
            yield dict(path=p, speaker=f"{actor:02d}", emotion=RAVDESS[parts[2]],
                       gender="M" if actor % 2 else "F", method="acted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cremad", "ravdess"], default=None if config.DATASET == "iemocap" else config.DATASET)
    ap.add_argument("--root", default=str(config.AUDIO_ROOT))
    ap.add_argument("--out", default=str(config.CSV))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    if a.dataset is None:
        raise SystemExit("config.DATASET is 'iemocap': it uses the provided CSV, so make_csv.py is not needed.")

    rows = list((rows_cremad if a.dataset == "cremad" else rows_ravdess)(a.root))
    if not rows:
        raise SystemExit(f"no matching .wav files under {a.root} -- check --root and --dataset")
    df = pd.DataFrame(rows)
    df["path"] = [os.path.relpath(p, a.root).replace("\\", "/") for p in df.path]

    spk = sorted(df.speaker.unique())
    fold = {s: i % a.folds + 1 for i, s in enumerate(spk)}  # round-robin over sorted actor ids
    df["session"] = df.speaker.map(fold)
    df["n_annotators"], df["agreement"] = 0, 4  # unused here; keeps the IEMOCAP schema
    df = df[["session", "method", "gender", "emotion", "n_annotators", "agreement", "path", "speaker"]]
    df.to_csv(a.out, index=False)
    print(f"{len(df)} clips, {len(spk)} speakers -> {a.out}")
    print(pd.crosstab(df.session, df.emotion))


if __name__ == "__main__":
    main()
