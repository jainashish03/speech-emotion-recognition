"""
Predict emotion from any audio file using a trained WavLM or Transformer model.

Usage:
  python infer.py path/to/audio.wav                          # uses WavLM by default
  python infer.py path/to/audio.wav --model transformer      # use Transformer instead
  python infer.py path/to/audio.wav --model wavlm --fold 1   # use a specific fold checkpoint

Examples:
  python infer.py ~/Downloads/IEMOCAP_full_release/Session5/sentences/wav/Ses05F_impro04/Ses05F_impro04_M029.wav
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config

LABELS = ["neutral", "happy", "sad", "angry"]
EMOJIS = {"neutral": "😐", "happy": "😄", "sad": "😢", "angry": "😡"}
SR = 16000
N_MELS = 64


# ──────────────────────────────────────────── models (copied from train.py)

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, h, mask):
        a = self.w(h).squeeze(-1).float().masked_fill(~mask, -1e4).softmax(-1)
        return (h * a.unsqueeze(-1).to(h.dtype)).sum(1)


def head(d, n_cls, drop):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(d, d // 2), nn.GELU(),
                         nn.Dropout(drop), nn.Linear(d // 2, n_cls))


def len_mask(L, T):
    return torch.arange(T, device=L.device)[None, :] < L[:, None]


class SinPos(nn.Module):
    def __init__(self, d, n=4000):
        super().__init__()
        pos, i = torch.arange(n)[:, None], torch.arange(0, d, 2)[None]
        pe = torch.zeros(n, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / 10000 ** (i / d)), torch.cos(pos / 10000 ** (i / d))
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)].to(x.dtype)


class TransformerSER(nn.Module):
    def __init__(self, D, d=256, heads=4, layers=4, ff=1024, drop=0.1, n_cls=4):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.sub = nn.Sequential(nn.Conv1d(D, d, 3, 2, 1), nn.GELU(), nn.Conv1d(d, d, 3, 2, 1), nn.GELU())
        self.pos = SinPos(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.pool = AttnPool(d)
        self.head = head(d, n_cls, 0.3)

    def forward(self, x, L):
        h = self.sub(self.norm(x).transpose(1, 2)).transpose(1, 2)
        L2 = (L - 1) // 2 + 1
        L2 = (L2 - 1) // 2 + 1
        mask = len_mask(L2, h.size(1))
        h = self.enc(self.pos(h), src_key_padding_mask=~mask)
        return self.head(self.pool(h, mask))


class WavLMSER(nn.Module):
    def __init__(self, name="microsoft/wavlm-base-plus", n_cls=4, drop=0.2):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name, mask_time_prob=0.05, layerdrop=0.0)
        H, nl = self.enc.config.hidden_size, self.enc.config.num_hidden_layers + 1
        self.lw = nn.Parameter(torch.zeros(nl))
        self.proj = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Dropout(drop))
        self.pool = AttnPool(256)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(256, n_cls))

    def forward(self, w, L):
        hs = self.enc(w, output_hidden_states=True).hidden_states
        h = (torch.stack(hs, 0) * self.lw.softmax(0)[:, None, None, None].to(hs[0].dtype)).sum(0)
        Lo = self.enc._get_feat_extract_output_lengths(L).clamp(max=h.size(1))
        return self.head(self.pool(self.proj(h), len_mask(Lo, h.size(1))))


# ──────────────────────────────────────────── audio loading

def trim_silence(w, top_db=22, frame_len=512, hop_len=128):
    """Trim leading and trailing non-speech so dead air doesn't bias the model towards Sad."""
    if len(w) <= frame_len:
        return w
    num_frames = (len(w) - frame_len) // hop_len + 1
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        seg = w[i * hop_len: i * hop_len + frame_len]
        rms[i] = np.sqrt(np.mean(seg * seg))
    peak_rms = float(np.max(rms))
    if peak_rms < 1e-4:
        return w
    thresh = peak_rms * (10 ** (-top_db / 20.0))
    active = np.where(rms > thresh)[0]
    if len(active) == 0:
        return w
    buf_samples = int(0.12 * SR)
    st = max(0, active[0] * hop_len - buf_samples)
    en = min(len(w), (active[-1] + 1) * hop_len + buf_samples)
    trimmed = w[st:en]
    return trimmed if len(trimmed) >= 1600 else w


def load_audio_wavlm(source, max_sec=6.0):
    """Load raw waveform for WavLM from file path, bytes, or file-like object."""
    import io
    import soundfile as sf
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    w, sr = sf.read(source, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)

    # Trim non-speech boundaries so silence doesn't falsely trigger Sad
    w = trim_silence(w)

    # Ensure minimum audio length (0.1s at 16kHz = 1600 samples)
    if len(w) == 0:
        w = np.zeros(1600, dtype=np.float32)
    elif len(w) < 1600:
        w = np.pad(w, (0, 1600 - len(w)))
    mx = int(max_sec * SR)
    if len(w) > mx:
        w = w[(len(w) - mx) // 2: (len(w) - mx) // 2 + mx]
    w = (w - w.mean()) / (w.std() + 1e-6)
    return torch.from_numpy(w).unsqueeze(0)  # (1, T)


class EmotionPredictor:
    """Pre-loaded in-memory inference engine supporting WavLM Impro, WavLM Full, and Multimodal."""
    def __init__(self, model="wavlm_impro", device="auto", folds=None):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else \
                     "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_type = model

        if model == "multimodal":
            runs_audio = config.RUNS_DIR / "wavlm"
            runs_text = config.RUNS_DIR / "roberta"
            self.fold_nums = folds if folds else sorted(int(p.stem.replace("fold", "")) for p in runs_audio.glob("fold*.pt"))
            
            print(f"Loading Multimodal SER (WavLM + RoBERTa) on {self.device}...")
            self.model_audio = WavLMSER().to(self.device)
            self.model_audio.eval()
            self.audio_states = [torch.load(runs_audio / f"fold{f}.pt", map_location=self.device, weights_only=True) for f in self.fold_nums]

            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.model_text = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4).to(self.device)
            self.model_text.eval()
            self.text_states = [torch.load(runs_text / f"fold{f}.pt", map_location=self.device, weights_only=True) for f in self.fold_nums]
            print(f"✓ Cached 5 audio + 5 text fold checkpoints for Multimodal inference.")
        else:
            folder_name = "wavlm_impro" if model == "wavlm_impro" else "wavlm"
            runs_path = config.RUNS_DIR / folder_name
            self.fold_nums = folds if folds else sorted(int(p.stem.replace("fold", "")) for p in runs_path.glob("fold*.pt"))

            print(f"Loading {model} on {self.device} from {runs_path.name}...")
            self.model_audio = WavLMSER().to(self.device)
            self.model_audio.eval()
            self.audio_states = [torch.load(runs_path / f"fold{f}.pt", map_location=self.device, weights_only=True) for f in self.fold_nums]
            print(f"✓ Cached {len(self.audio_states)} fold checkpoints for {model}.")

    def predict(self, audio_source, text=None):
        x = load_audio_wavlm(audio_source).to(self.device)
        L = torch.tensor([x.shape[1]], device=self.device)

        logits_list_audio = []
        with torch.no_grad():
            for st in self.audio_states:
                self.model_audio.load_state_dict(st, strict=False)
                self.model_audio.eval()
                lg = self.model_audio(x, L).squeeze(0).cpu().numpy()
                logits_list_audio.append(lg)

        # Average logits across folds before softmax (deep learning ensemble standard)
        avg_audio_logits = np.mean(logits_list_audio, axis=0)
        # Apply balanced logit calibration offset (-0.35 on Sad) to eliminate theatrical acting bias
        avg_audio_logits[2] -= 0.35
        avg_audio_probs = torch.tensor(avg_audio_logits / 1.15).softmax(-1).numpy()

        if self.model_type == "multimodal" and text and text.strip():
            inputs = self.tokenizer(text.strip(), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
            inp_ids = inputs["input_ids"].to(self.device)
            att_mask = inputs["attention_mask"].to(self.device)

            logits_list_text = []
            with torch.no_grad():
                for st in self.text_states:
                    self.model_text.load_state_dict(st, strict=False)
                    self.model_text.eval()
                    lg = self.model_text(input_ids=inp_ids, attention_mask=att_mask).logits.squeeze(0).cpu().numpy()
                    logits_list_text.append(lg)
            avg_text_logits = np.mean(logits_list_text, axis=0)
            avg_text_probs = torch.tensor(avg_text_logits).softmax(-1).numpy()

            # When text transcript is explicitly provided, give text 70% weight so word semantics dominate
            final_probs = 0.30 * avg_audio_probs + 0.70 * avg_text_probs
        else:
            final_probs = avg_audio_probs

        if self.device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            torch.mps.empty_cache()

        pred_idx = int(final_probs.argmax())
        pred_label = LABELS[pred_idx]
        conf = float(final_probs[pred_idx] * 100)

        probs_dict = {label: round(float(final_probs[i] * 100), 1) for i, label in enumerate(LABELS)}
        audio_probs_dict = {label: round(float(avg_audio_probs[i] * 100), 1) for i, label in enumerate(LABELS)}
        text_probs_dict = {label: round(float(avg_text_probs[i] * 100), 1) for i, label in enumerate(LABELS)} if (self.model_type == "multimodal" and text and text.strip()) else None

        return {
            "status": "ok",
            "model": self.model_type,
            "pred": pred_label,
            "emoji": EMOJIS[pred_label],
            "conf": round(conf, 1),
            "probs": probs_dict,
            "audio_probs": audio_probs_dict,
            "text_probs": text_probs_dict,
            "folds": len(self.audio_states),
            "has_text": bool(text and text.strip())
        }


def load_audio_transformer(path, data_dir, max_frames=800):
    """Load pre-extracted features for the Transformer model."""
    import pandas as pd
    feats = np.load(f"{data_dir}/feats.npy", mmap_mode="r")
    offsets = np.load(f"{data_dir}/offsets.npy")
    meta = pd.read_csv(f"{data_dir}/meta.csv")
    wav_path = str(Path(path).resolve())
    row = meta[meta.wav == wav_path]
    if row.empty:
        sys.exit(f"\n  Error: '{path}' not found in meta.csv.\n"
                 f"  The Transformer model requires pre-extracted features.\n"
                 f"  Use --model wavlm instead, which works on any .wav file.")
    idx = row.idx.values[0]
    x = np.array(feats[offsets[idx]:offsets[idx + 1]], dtype=np.float32)
    if len(x) > max_frames:
        st = (len(x) - max_frames) // 2
        x = x[st:st + max_frames]
    T = torch.from_numpy(x).unsqueeze(0)    # (1, T, D)
    L = torch.tensor([T.shape[1]])
    return T, L


def record_audio_mic(duration=4.0, sr=16000, out_path="my_voice.wav"):
    """Record audio from user's microphone."""
    import time
    import sounddevice as sd
    import soundfile as sf
    print(f"\n🎙️  Microphone Recorder")
    print(f"   Duration: {duration:.1f}s | Sample Rate: {sr} Hz")
    if sys.stdin.isatty():
        try:
            input("   👉 Press [Enter] when ready to speak... ")
        except (EOFError, IOError):
            pass
    else:
        print("   Auto-starting in 1s...")
        time.sleep(1)
    print(f"🔴 Recording for {duration:.1f} seconds... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording finished!")

    # Remove DC offset & apply smart gain normalization
    if len(audio) > 0:
        audio = audio - np.mean(audio)
        max_amp = float(np.max(np.abs(audio)))
        if max_amp > 1e-4 and max_amp < 0.7:
            gain = min(20.0, 0.85 / max_amp)
            audio = np.clip(audio * gain, -1.0, 1.0)
            print(f"   🔊 Boosted low-level microphone audio (gain: {gain:.1f}x, peak: {np.max(np.abs(audio)):.2f})")
        elif max_amp <= 1e-4:
            print("⚠️  Warning: Recorded audio is silent or near-zero. Check microphone input volume.")
    else:
        max_amp = 0.0

    sf.write(out_path, audio, sr)
    print(f"💾 Saved audio clip to: {out_path} (raw peak: {max_amp:.3f})")
    return Path(out_path).resolve()


# ──────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("audio", nargs="?", default=None, help="Path to any .wav audio file (omit to record live)")
    ap.add_argument("--record", action="store_true", help="Record audio directly from your microphone")
    ap.add_argument("--seconds", type=float, default=4.0, help="Duration of microphone recording in seconds (default: 4.0)")
    ap.add_argument("--text", default=None, help="Spoken transcript (enables Multimodal WavLM+RoBERTa prediction)")
    ap.add_argument("--model", choices=["wavlm", "transformer", "multimodal", "wavlm_impro", "auto"], default="wavlm")
    ap.add_argument("--fold", type=int, default=None,
                    help="Which fold checkpoint to use (1-5). Default: averages all available folds.")
    ap.add_argument("--data", default=str(config.DATA_DIR),
                    help="Path to prepared data dir (needed for --model transformer)")
    ap.add_argument("--runs", default=str(config.RUNS_DIR))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.record or args.audio is None:
        audio_path = record_audio_mic(duration=args.seconds)
    else:
        audio_path = Path(args.audio).expanduser().resolve()
        if not audio_path.exists():
            # Check common fallback locations like parent directory or ~/Downloads
            cand1 = Path.cwd().parent / args.audio
            cand2 = Path.home() / "Downloads" / args.audio
            if cand1.exists():
                audio_path = cand1.resolve()
            elif cand2.exists():
                audio_path = cand2.resolve()
            else:
                sys.exit(f"Audio file not found: {audio_path}")

    # Only load text if explicitly requested or running multimodal
    text_input = args.text
    if args.model == "multimodal" and text_input is None:
        mm_meta_path = Path(args.data) / "meta_multimodal.csv"
        if mm_meta_path.exists():
            import pandas as pd
            df_mm = pd.read_csv(mm_meta_path)
            row = df_mm[df_mm.wav == str(audio_path)]
            if not row.empty and pd.notna(row.text.values[0]):
                text_input = str(row.text.values[0])

    if args.model == "auto":
        args.model = "wavlm"

    dev_str = args.device
    if dev_str == "auto":
        dev_str = "mps" if torch.backends.mps.is_available() else \
                  "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_str)

    # ── Predict with Audio (WavLM or Transformer)
    runs_path = Path(args.runs)
    if (runs_path / "fold1.pt").exists():
        model_dir_audio = runs_path
        audio_model_name = "wavlm"
    elif args.model == "wavlm" and (runs_path / "wavlm_impro" / "fold1.pt").exists():
        # Prefer the 73.2% champion model by default
        model_dir_audio = runs_path / "wavlm_impro"
        audio_model_name = "wavlm"
    elif (runs_path / args.model / "fold1.pt").exists():
        model_dir_audio = runs_path / args.model
        audio_model_name = "transformer" if args.model == "transformer" else "wavlm"
    else:
        audio_model_name = "wavlm" if args.model in ["wavlm", "multimodal", "wavlm_impro"] else "transformer"
        model_dir_audio = runs_path / audio_model_name
    folds = [args.fold] if args.fold else sorted(int(p.stem.replace("fold", "")) for p in model_dir_audio.glob("fold*.pt"))

    print(f"\nDevice: {dev} | Model: {audio_model_name} ({model_dir_audio.name}) | Folds: {len(folds)}")
    if text_input and args.model == "multimodal":
        print(f"Spoken Text: \"{text_input}\"")

    if audio_model_name == "wavlm":
        audio_model = WavLMSER()
        x_audio = load_audio_wavlm(str(audio_path)).to(dev)
        L_audio = torch.tensor([x_audio.shape[1]], device=dev)
    else:
        D = np.load(f"{args.data}/feats.npy", mmap_mode="r").shape[1]
        audio_model = TransformerSER(D)
        x_audio, L_audio = load_audio_transformer(str(audio_path), args.data)
        x_audio, L_audio = x_audio.to(dev), L_audio.to(dev)
    audio_model.to(dev)

    audio_probs_list = []
    for f in folds:
        ckpt = model_dir_audio / f"fold{f}.pt"
        st = torch.load(ckpt, map_location=dev, weights_only=True)
        audio_model.load_state_dict(st, strict=False)
        audio_model.eval()
        with torch.no_grad():
            lg = audio_model(x_audio, L_audio)
            audio_probs_list.append(lg.float().softmax(-1).squeeze(0).cpu().numpy())
        if dev_str == "mps":
            torch.mps.empty_cache()
    avg_audio_probs = np.mean(audio_probs_list, axis=0)

    # ── Predict with Text if Multimodal
    if args.model == "multimodal" and text_input is not None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained("roberta-base")
        inputs = tok(text_input, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        inp_ids = inputs["input_ids"].to(dev)
        att_mask = inputs["attention_mask"].to(dev)

        model_dir_text = Path(args.runs) / "roberta"
        text_folds = [args.fold] if args.fold else sorted(int(p.stem.replace("fold", "")) for p in model_dir_text.glob("fold*.pt"))
        text_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4).to(dev)

        text_probs_list = []
        for f in text_folds:
            ckpt = model_dir_text / f"fold{f}.pt"
            st = torch.load(ckpt, map_location=dev, weights_only=True)
            text_model.load_state_dict(st, strict=False)
            text_model.eval()
            with torch.no_grad():
                lg = text_model(input_ids=inp_ids, attention_mask=att_mask).logits
                text_probs_list.append(lg.float().softmax(-1).squeeze(0).cpu().numpy())
            if dev_str == "mps":
                torch.mps.empty_cache()
        avg_text_probs = np.mean(text_probs_list, axis=0)

        # Fused combination: 0.45 Audio + 0.55 Text
        final_probs = 0.45 * avg_audio_probs + 0.55 * avg_text_probs
    else:
        final_probs = avg_audio_probs

    pred_idx = final_probs.argmax()
    pred_label = LABELS[pred_idx]

    # ── print results
    print(f"\n{'─'*44}")
    print(f"  File : {audio_path.name}")
    print(f"  {'─'*40}")
    print(f"  Prediction: {EMOJIS[pred_label]}  {pred_label.upper()}  ({final_probs[pred_idx]*100:.1f}% confidence)")
    print(f"  {'─'*40}")
    print(f"  {'Emotion':<10}  {'Confidence':>10}   Bar")
    print(f"  {'─'*40}")
    for i, (label, p) in enumerate(zip(LABELS, final_probs)):
        bar = "█" * int(p * 30)
        marker = " ◄" if i == pred_idx else ""
        print(f"  {EMOJIS[label]} {label:<9}  {p*100:>8.1f}%   {bar}{marker}")
    print(f"{'─'*44}\n")


if __name__ == "__main__":
    main()

