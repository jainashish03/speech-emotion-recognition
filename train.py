"""
Train SER models on IEMOCAP (4 classes) with leave-one-session-out cross-validation.

  python train.py --model bilstm       --data data          # RNN on log-mel + prosodic DSP frames
  python train.py --model transformer  --data data          # Transformer encoder, same frames
  python train.py --model wavlm        --data data          # fine-tune WavLM on raw waveforms (needs `transformers`)

Run prepare_data.py first. For each held-out session: 10% of the training utterances (stratified) are used
for early stopping / model selection, so the test session is never used to pick epochs.
Reports UA (unweighted accuracy = mean per-class recall), WA, macro-F1, confusion matrix.
Outputs in runs/<name>/: results.json, predictions.csv, fold*.pt, dashboard_results.json
"""
import argparse
import contextlib
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import config  # default paths live in config.py

LABELS = ["neutral", "happy", "sad", "angry"]
N_MELS = 64
SR = 16000


# ----------------------------------------------------------------------------- data
def augment(x):
    """SpecAugment-style masking + noise on speaker-normalised frames (0 == speaker mean)."""
    T = len(x)
    for _ in range(2):  # frequency masks on the mel part only
        w = np.random.randint(0, 9)
        f0 = np.random.randint(0, max(1, N_MELS - w))
        x[:, f0:f0 + w] = 0
    for _ in range(2):  # time masks on every channel
        w = np.random.randint(0, max(2, int(0.08 * T)))
        t0 = np.random.randint(0, max(1, T - w))
        x[t0:t0 + w] = 0
    x += np.random.randn(*x.shape).astype(np.float32) * 0.05
    return x


class SeqDS(Dataset):
    def __init__(self, feats_path, offsets, meta, train, max_frames=800):
        self.fp, self.f, self.o, self.train, self.mx = feats_path, None, offsets, train, max_frames
        self.ids, self.y = meta.idx.values, meta.label.values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.f is None:  # opened lazily so DataLoader workers (incl. Windows spawn) don't copy the array
            self.f = np.load(self.fp, mmap_mode="r")
        k = self.ids[i]
        x = np.array(self.f[self.o[k]:self.o[k + 1]], dtype=np.float32)
        if len(x) > self.mx:
            st = np.random.randint(0, len(x) - self.mx + 1) if self.train else (len(x) - self.mx) // 2
            x = x[st:st + self.mx]
        if self.train:
            x = augment(x)
        return torch.from_numpy(x), int(self.y[i])


def collate_seq(b):
    xs, ys = zip(*b)
    L = torch.tensor([len(x) for x in xs])
    return pad_sequence(xs, batch_first=True), L, torch.tensor(ys)


def seed_worker(w):
    np.random.seed((torch.initial_seed() + w) % 2**32)


class WavDS(Dataset):
    def __init__(self, meta, train, max_sec=6.0):
        self.paths, self.y, self.train, self.mx = meta.wav.values, meta.label.values, train, int(max_sec * SR)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import soundfile as sf
        w, sr = sf.read(self.paths[i], dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            g = math.gcd(sr, SR)
            w = resample_poly(w, SR // g, sr // g).astype(np.float32)

        # Audio augmentations for training
        if self.train:
            # 1. Speed perturbation (0.9x to 1.1x) with 50% probability
            if np.random.rand() < 0.5:
                speed = np.random.uniform(0.9, 1.1)
                new_len = max(int(len(w) / speed), 1600)
                w = np.interp(np.linspace(0, len(w), new_len, endpoint=False), np.arange(len(w)), w).astype(np.float32)
            # 2. Random gain scaling (0.8x to 1.2x)
            w = w * np.random.uniform(0.8, 1.2)

        if len(w) > self.mx:
            st = np.random.randint(0, len(w) - self.mx + 1) if self.train else (len(w) - self.mx) // 2
            w = w[st:st + self.mx]

        w = (w - w.mean()) / (w.std() + 1e-6)

        if self.train:
            # 3. Additive subtle noise
            w = w + np.random.randn(len(w)).astype(np.float32) * np.random.uniform(0.002, 0.015)
            # 4. Random time masking (50 - 100 ms) with 30% probability
            if np.random.rand() < 0.3 and len(w) > 3200:
                mask_len = int(np.random.uniform(0.05, 0.10) * SR)
                mask_st = np.random.randint(0, len(w) - mask_len)
                w[mask_st:mask_st + mask_len] = 0.0

        return torch.from_numpy(w.astype(np.float32)), int(self.y[i])


# ----------------------------------------------------------------------------- models
def len_mask(L, T):
    return torch.arange(T, device=L.device)[None, :] < L[:, None]


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, h, mask):
        a = self.w(h).squeeze(-1).float().masked_fill(~mask, -1e4).softmax(-1)
        return (h * a.unsqueeze(-1).to(h.dtype)).sum(1)


def head(d, n_cls, drop):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, n_cls))


class BiLSTMAtt(nn.Module):
    """Frame features -> projection -> 2-layer BiLSTM -> attention pooling -> classifier."""

    def __init__(self, D, hid=192, layers=2, drop=0.3, n_cls=4):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Sequential(nn.Linear(D, hid), nn.GELU(), nn.Dropout(drop))
        self.rnn = nn.LSTM(hid, hid, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.pool = AttnPool(2 * hid)
        self.head = head(2 * hid, n_cls, drop)

    def forward(self, x, L):
        h = self.proj(self.norm(x))
        pk = nn.utils.rnn.pack_padded_sequence(h, L.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.rnn(pk)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.size(1))
        return self.head(self.pool(o, len_mask(L, x.size(1))))


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
    """Conv subsampling (x4) -> sinusoidal positions -> pre-norm Transformer encoder -> attention pooling."""

    def __init__(self, D, d=256, heads=4, layers=4, ff=1024, drop=0.1, n_cls=4):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.sub = nn.Sequential(nn.Conv1d(D, d, 3, 2, 1), nn.GELU(), nn.Conv1d(d, d, 3, 2, 1), nn.GELU())
        self.pos = SinPos(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu", batch_first=True, norm_first=True)
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
    """Pretrained WavLM (CNN frozen) -> learned weighted sum of layers -> attention pooling -> classifier."""

    def __init__(self, name, n_cls=4, drop=0.2):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name, mask_time_prob=0.05, layerdrop=0.0)
        self.enc.freeze_feature_encoder()
        H, nl = self.enc.config.hidden_size, self.enc.config.num_hidden_layers + 1
        self.lw = nn.Parameter(torch.zeros(nl))
        self.proj = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Dropout(drop))
        self.pool = AttnPool(256)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(256, n_cls))

    def forward(self, w, L):
        hs = self.enc(w, output_hidden_states=True).hidden_states  # base models: no attention_mask, zero-padded
        h = (torch.stack(hs, 0) * self.lw.softmax(0)[:, None, None, None].to(hs[0].dtype)).sum(0)
        Lo = self.enc._get_feat_extract_output_lengths(L).clamp(max=h.size(1))
        return self.head(self.pool(self.proj(h), len_mask(Lo, h.size(1))))


def collate_wav(b):
    ws, ys = zip(*b)
    L = torch.tensor([len(w) for w in ws])
    return pad_sequence(ws, batch_first=True), L, torch.tensor(ys)


# ----------------------------------------------------------------------------- training
def metrics(y, p):
    return dict(ua=float(recall_score(y, p, average="macro", zero_division=0)), wa=float(accuracy_score(y, p)),
                f1=float(f1_score(y, p, average="macro", zero_division=0)))


def amp_ctx(dev):
    if dev.type != "cuda":
        return contextlib.nullcontext(), False
    bf16 = torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16), not bf16


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    ctx, _ = amp_ctx(dev)
    out = []
    for x, L, _y in dl:
        with ctx:
            lg = model(x.to(dev), L.to(dev))
        out.append(lg.float().softmax(-1).cpu())
        if dev.type == "mps":
            torch.mps.empty_cache()
    return torch.cat(out).numpy()


def make_sched(opt, total, warm=0.1):
    w = max(1, int(warm * total))

    def f(s):
        return (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, total - w)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def build(args, D):
    if args.model == "bilstm":
        return BiLSTMAtt(D)
    if args.model == "transformer":
        return TransformerSER(D)
    return WavLMSER(args.wavlm_name)


def run_fold(args, meta, feats_path, D, offsets, test_session, dev):
    tr_all = meta[meta.session != test_session]
    te = meta[meta.session == test_session]
    tr, va = train_test_split(tr_all, test_size=0.1, stratify=tr_all.label, random_state=args.seed)
    if args.model == "wavlm":
        mk = lambda m, t: DataLoader(WavDS(m, t), args.bs, shuffle=t, collate_fn=collate_wav, num_workers=args.workers,
                                     drop_last=t and len(m) > args.bs, pin_memory=dev.type == "cuda", worker_init_fn=seed_worker)
    else:
        mk = lambda m, t: DataLoader(SeqDS(feats_path, offsets, m, t, args.max_frames), args.bs, shuffle=t,
                                     collate_fn=collate_seq, num_workers=args.workers, drop_last=t and len(m) > args.bs,
                                     pin_memory=dev.type == "cuda", worker_init_fn=seed_worker)
    tr_dl, va_dl, te_dl = mk(tr, True), mk(va, False), mk(te, False)

    model = build(args, D).to(dev)
    cnt = np.bincount(tr.label.values, minlength=4)
    cw = torch.tensor(len(tr) / (4 * cnt), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    if args.model == "wavlm":
        bb = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
        hd = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": hd, "lr": args.lr}], weight_decay=1e-2)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(tr_dl) / args.accum)
    sched = make_sched(opt, args.epochs * steps_per_epoch)
    ctx, use_scaler = amp_ctx(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    params = [p for p in model.parameters() if p.requires_grad]

    best, best_state, bad, hist = -1, None, 0, []
    yv = va.label.values
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for i, (x, L, y) in enumerate(tr_dl):
            with ctx:
                lg = model(x.to(dev, non_blocking=True), L.to(dev))
            loss = crit(lg.float(), y.to(dev)) / args.accum
            scaler.scale(loss).backward()
            tot += loss.item() * args.accum
            if (i + 1) % args.accum == 0 or i + 1 == len(tr_dl):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if dev.type == "mps":
                    torch.mps.empty_cache()
        pv = predict(model, va_dl, dev).argmax(1)
        m = metrics(yv, pv)
        hist.append(dict(epoch=ep + 1, loss=tot / len(tr_dl), val_ua=m["ua"], val_wa=m["wa"]))
        flag = ""
        if m["ua"] > best:
            best, bad, flag = m["ua"], 0, " *"
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"  [S{test_session}] ep {ep + 1:02d} loss {tot / len(tr_dl):.3f} val UA {m['ua']:.3f} "
              f"WA {m['wa']:.3f} ({time.time() - t0:.0f}s){flag}", flush=True)
        if bad >= args.patience:
            print("  early stop")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{args.out}/{args.name}/fold{test_session}.pt")
    probs = predict(model, te_dl, dev)
    m = metrics(te.label.values, probs.argmax(1))
    print(f"==> held-out Session{test_session}: UA {m['ua']:.3f}  WA {m['wa']:.3f}  F1 {m['f1']:.3f}  (best val UA {best:.3f})")
    return te, probs, m, hist, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bilstm", "transformer", "wavlm"], required=True)
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.RUNS_DIR))
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--wavlm-name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=0, help="debug: subsample N utterances")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--impro-only", action="store_true", help="Standard IEMOCAP benchmark: train/test on improvised speech only")
    a = ap.parse_args()
    d = dict(bilstm=(40, 32, 1, 1e-3), transformer=(60, 32, 1, 5e-4), wavlm=(12, 4, 8, 1e-3))[a.model]
    a.epochs, a.bs, a.accum, a.lr = [v if v is not None else dv for v, dv in zip((a.epochs, a.bs, a.accum, a.lr), d)]
    if a.impro_only and a.name is None:
        a.name = f"{a.model}_impro"
    else:
        a.name = a.name or a.model
    os.makedirs(f"{a.out}/{a.name}", exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       "cpu" if a.device == "auto" else a.device)
    print(f"device: {dev} | model: {a.model} | epochs {a.epochs} bs {a.bs}x{a.accum} lr {a.lr}")

    meta = pd.read_csv(f"{a.data}/meta.csv")
    if a.impro_only:
        meta = meta[meta.method == "impro"].reset_index(drop=True)
        print(f"Filtered to standard IEMOCAP improvised benchmark: {len(meta)} utterances")
    if a.limit:
        meta = meta.sample(a.limit, random_state=0).reset_index(drop=True)
    feats_path = f"{a.data}/feats.npy"
    D = np.load(feats_path, mmap_mode="r").shape[1]
    offsets = np.load(f"{a.data}/offsets.npy")

    parts, folds, hist = [], {}, {}
    for s in a.folds:
        te, probs, m, h, bv = run_fold(a, meta, feats_path, D, offsets, s, dev)
        te = te.copy()
        te["pred"] = probs.argmax(1)
        for i, n in enumerate(LABELS):
            te[f"p_{n}"] = probs[:, i]
        parts.append(te)
        folds[s], hist[s] = dict(m, best_val_ua=bv), h

    P = pd.concat(parts)
    P.drop(columns=["wav"]).to_csv(f"{a.out}/{a.name}/predictions.csv", index=False)
    y, p = P.label.values, P.pred.values
    overall = metrics(y, p)
    cm = confusion_matrix(y, p, labels=range(4))
    ua = [f["ua"] for f in folds.values()]
    print(f"\nOVERALL ({len(a.folds)} folds)  UA {overall['ua']:.3f}  WA {overall['wa']:.3f}  F1 {overall['f1']:.3f}  "
          f"| per-fold UA {np.mean(ua):.3f} +- {np.std(ua):.3f}")
    print("confusion (rows=true, cols=pred)", LABELS)
    print(cm)
    res = dict(model=a.model, args=vars(a), overall=overall, per_fold=folds, ua_mean=float(np.mean(ua)),
               ua_std=float(np.std(ua)), confusion=cm.tolist(), labels=LABELS, history=hist)
    json.dump(res, open(f"{a.out}/{a.name}/results.json", "w"), indent=1)
    # compact file the dashboard can import
    json.dump(dict(model=a.model, ua=overall["ua"], wa=overall["wa"], f1=overall["f1"], per_fold_ua=ua,
                   confusion=cm.tolist(), labels=LABELS), open(f"{a.out}/{a.name}/dashboard_results.json", "w"))


if __name__ == "__main__":
    main()
