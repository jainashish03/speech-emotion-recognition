"""
Parse IEMOCAP dialog transcriptions and attach the text for every utterance.
Outputs: data_iemocap/meta_multimodal.csv
"""
import glob
import os
import re
from pathlib import Path
import pandas as pd
import config

def main():
    root = config.audio_root()
    trans_files = sorted(glob.glob(f"{root}/Session*/dialog/transcriptions/*.txt"))
    print(f"Found {len(trans_files)} transcription files under {root}")

    trans_dict = {}
    for tf in trans_files:
        with open(tf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # Matches: Ses01F_script02_1_F000 [015.1400-017.2100]: Fine.
                m = re.match(r"^([A-Za-z0-9_]+)\s*\[[0-9\.\-]+\]:\s*(.*)$", line)
                if m:
                    uid = m.group(1).strip()
                    text = m.group(2).strip()
                    # Clean up multiple whitespace
                    text = re.sub(r"\s+", " ", text)
                    trans_dict[uid] = text

    print(f"Parsed {len(trans_dict)} total transcribed lines.")

    meta_path = config.DATA_DIR / "meta.csv"
    df = pd.read_csv(meta_path)
    
    texts = []
    missing = 0
    for wav_path in df.wav:
        uid = Path(wav_path).stem
        if uid in trans_dict:
            texts.append(trans_dict[uid])
        else:
            texts.append("")
            missing += 1

    df["text"] = texts
    out_path = config.DATA_DIR / "meta_multimodal.csv"
    df.to_csv(out_path, index=False)
    print(f"Attached text to {len(df) - missing}/{len(df)} utterances (missing: {missing}).")
    print(f"Saved to: {out_path}")

    # Show a few samples
    print("\nSample paired utterances:")
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"  [{row.session}{row.gender} | {row.label_name.upper():7s}] {row.text}")

if __name__ == "__main__":
    main()

