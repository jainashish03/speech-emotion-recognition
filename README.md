# SER project (Speech Emotion Recognition)

Prosodic DSP + RNN / Transformer / WavLM on IEMOCAP (also works with CREMA-D and RAVDESS).

## Quick start
1. `pip install -r requirements.txt`   (for GPU use the PyTorch build from pytorch.org)
2. Put your audio in `audio/` (see `audio/PUT_AUDIO_HERE.txt`). Edit `config.py` only if your audio is elsewhere
   or you use CREMA-D/RAVDESS (`DATASET = "cremad"`).
3. `python run.py`            -> checks everything and tells you what is missing
4. `python run.py all --limit 200 --epochs 2 --folds 1`   -> quick smoke test
5. `python run.py all`        -> full run (prepare + train every model in `config.MODELS`)

Other commands: `prepare`, `baseline` (SVM, no GPU needed), `train`, `dashboard`.
Windows: double-click `run.bat` (or `run.bat all`).

## Files
config.py (paths, edit this) | run.py (entry point) | make_csv.py (CREMA-D/RAVDESS labels) |
prepare_data.py (features) | train.py (BiLSTM / Transformer / WavLM) | ser_pipeline.py (SVM baseline) |
dashboard/ser_dashboard.html (web dashboard) | iemocap_full_dataset.csv (IEMOCAP labels)

Results: `runs_<dataset>/<model>/results.json`, `predictions.csv`, `dashboard_results.json`.
