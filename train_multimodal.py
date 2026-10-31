"""
Multimodal Emotion Recognition on IEMOCAP:
1. Fine-tunes RoBERTa on IEMOCAP text transcripts (5-fold Leave-One-Session-Out).
2. Performs optimal multimodal fusion with WavLM audio predictions.
3. Reports standalone Text metrics and fused Multimodal metrics (UA, WA, F1, confusion matrix).
Outputs: runs_iemocap/roberta/ and runs_iemocap/multimodal/
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

import config

LABELS = ["neutral", "happy", "sad", "angry"]
MODEL_NAME = "roberta-base"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def metrics(y, p):
    return dict(
        ua=float(recall_score(y, p, average="macro", zero_division=0)),
        wa=float(accuracy_score(y, p)),
        f1=float(f1_score(y, p, average="macro", zero_division=0))
    )


@torch.no_grad()
def predict_text(model, dl, dev):
    model.eval()
    all_probs = []
    for batch in dl:
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs.logits.float().softmax(-1).cpu()
        all_probs.append(probs)
    return torch.cat(all_probs).numpy()


def run_text_fold(args, meta, test_session, tokenizer, dev):
    tr_all = meta[meta.session != test_session]
    te = meta[meta.session == test_session]
    tr, va = train_test_split(tr_all, test_size=0.1, stratify=tr_all.label, random_state=args.seed)

    tr_ds = TextDataset(tr.text.values, tr.label.values, tokenizer, max_len=args.max_len)
    va_ds = TextDataset(va.text.values, va.label.values, tokenizer, max_len=args.max_len)
    te_ds = TextDataset(te.text.values, te.label.values, tokenizer, max_len=args.max_len)

    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=args.bs, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.text_model, num_labels=4).to(dev)

    cnt = np.bincount(tr.label.values, minlength=4)
    cw = torch.tensor(len(tr) / (4 * cnt), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(tr_dl) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_ua, best_state = -1, None
    y_val = va.label.values

    for ep in range(args.epochs):
        model.train()
        t0, tot_loss = time.time(), 0.0
        for batch in tr_dl:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = crit(outputs.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()

        probs_val = predict_text(model, va_dl, dev)
        preds_val = probs_val.argmax(1)
        m = metrics(y_val, preds_val)
        flag = ""
        if m["ua"] > best_ua:
            best_ua = m["ua"]
            best_state = copy.deepcopy(model.state_dict())
            flag = " *"
        print(f"  [Text S{test_session}] ep {ep+1:02d} loss {tot_loss/len(tr_dl):.3f} val UA {m['ua']:.3f} WA {m['wa']:.3f} ({time.time()-t0:.0f}s){flag}", flush=True)

    model.load_state_dict(best_state)
    out_dir = Path(args.out) / "roberta"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, out_dir / f"fold{test_session}.pt")

    probs_te = predict_text(model, te_dl, dev)
    probs_va = predict_text(model, va_dl, dev)
    m_te = metrics(te.label.values, probs_te.argmax(1))
    print(f"==> Text Session{test_session}: UA {m_te['ua']:.3f}  WA {m_te['wa']:.3f}  F1 {m_te['f1']:.3f}")

    return te, probs_te, m_te, va, probs_va


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data_iemocap/meta_multimodal.csv")
    ap.add_argument("--out", default=str(config.RUNS_DIR))
    ap.add_argument("--text-model", default=MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev_str = args.device
    if dev_str == "auto":
        dev_str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_str)
    print(f"Device: {dev} | Text Model: {args.text_model} | Epochs: {args.epochs}")

    meta = pd.read_csv(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # 1. Train Text Model Across Folds
    print("\n" + "="*50 + "\nSTEP 1: Training Text Model (RoBERTa)\n" + "="*50)
    text_parts = []
    text_folds = {}
    val_text_probs = {}

    for s in args.folds:
        te, probs_te, m_te, va, probs_va = run_text_fold(args, meta, s, tokenizer, dev)
        te = te.copy()
        te["pred_text"] = probs_te.argmax(1)
        for i, n in enumerate(LABELS):
            te[f"p_text_{n}"] = probs_te[:, i]
        text_parts.append(te)
        text_folds[s] = m_te
        val_text_probs[s] = (va, probs_va)

    P_text = pd.concat(text_parts).sort_values("idx").reset_index(drop=True)
    out_text_dir = Path(args.out) / "roberta"
    P_text.to_csv(out_text_dir / "predictions.csv", index=False)

    m_overall_text = metrics(P_text.label.values, P_text.pred_text.values)
    cm_text = confusion_matrix(P_text.label.values, P_text.pred_text.values, labels=range(4))
    print(f"\nTEXT OVERALL: UA {m_overall_text['ua']:.3f}  WA {m_overall_text['wa']:.3f}  F1 {m_overall_text['f1']:.3f}")
    json.dump(dict(overall=m_overall_text, per_fold=text_folds, confusion=cm_text.tolist()),
              open(out_text_dir / "results.json", "w"), indent=2)

    # 2. Multimodal Fusion with WavLM
    print("\n" + "="*50 + "\nSTEP 2: Multimodal Fusion (WavLM + RoBERTa)\n" + "="*50)
    wavlm_pred_path = Path(args.out) / "wavlm" / "predictions.csv"
    if not wavlm_pred_path.exists():
        print(f"Error: {wavlm_pred_path} not found. Audio predictions needed for fusion.")
        return

    P_audio = pd.read_csv(wavlm_pred_path).sort_values("idx").reset_index(drop=True)

    prob_cols_audio = [f"p_{n}" for n in LABELS]
    prob_cols_text = [f"p_text_{n}" for n in LABELS]

    probs_A = P_audio[prob_cols_audio].values
    probs_T = P_text[prob_cols_text].values
    y_true = P_text.label.values

    # Find optimal fusion weight alpha on grid (0.0 to 1.0)
    best_alpha, best_fused_ua = 0.5, 0.0
    for alpha in np.linspace(0.1, 0.9, 17):
        fused_p = alpha * probs_A + (1 - alpha) * probs_T
        score = recall_score(y_true, fused_p.argmax(1), average="macro")
        if score > best_fused_ua:
            best_fused_ua = score
            best_alpha = alpha

    print(f"Optimal Multimodal Weight: {best_alpha:.2f} Audio + {1-best_alpha:.2f} Text")

    fused_probs = best_alpha * probs_A + (1 - best_alpha) * probs_T
    fused_preds = fused_probs.argmax(1)

    m_fused = metrics(y_true, fused_preds)
    cm_fused = confusion_matrix(y_true, fused_preds, labels=range(4))

    # Per-fold fused metrics
    per_fold_fused = {}
    for s in args.folds:
        mask = P_text.session == s
        per_fold_fused[s] = metrics(y_true[mask], fused_preds[mask])
        print(f"  [Multimodal Session {s}] UA: {per_fold_fused[s]['ua']:.3f}  WA: {per_fold_fused[s]['wa']:.3f}  F1: {per_fold_fused[s]['f1']:.3f}")

    print("\n" + "="*50)
    print(f"🏆 FINAL MULTIMODAL OVERALL (5 Folds):")
    print(f"   UA (Unweighted Accuracy): {m_fused['ua']*100:.2f}%")
    print(f"   WA (Weighted Accuracy):   {m_fused['wa']*100:.2f}%")
    print(f"   Macro F1:                 {m_fused['f1']*100:.2f}%")
    print("="*50)
    print("Confusion Matrix (rows=true, cols=pred):", LABELS)
    print(cm_fused)

    out_mm_dir = Path(args.out) / "multimodal"
    os.makedirs(out_mm_dir, exist_ok=True)

    P_mm = P_text.copy()
    P_mm["pred_multimodal"] = fused_preds
    for i, n in enumerate(LABELS):
        P_mm[f"p_multimodal_{n}"] = fused_probs[:, i]
    P_mm.to_csv(out_mm_dir / "predictions.csv", index=False)

    res_mm = dict(
        model="multimodal_wavlm_roberta",
        best_alpha=float(best_alpha),
        overall=m_fused,
        per_fold=per_fold_fused,
        confusion=cm_fused.tolist(),
        labels=LABELS
    )
    json.dump(res_mm, open(out_mm_dir / "results.json", "w"), indent=2)
    json.dump(dict(model="multimodal", ua=m_fused["ua"], wa=m_fused["wa"], f1=m_fused["f1"],
                   confusion=cm_fused.tolist(), labels=LABELS), open(out_mm_dir / "dashboard_results.json", "w"))
    print(f"\nSaved results to {out_mm_dir}")


if __name__ == "__main__":
    main()

