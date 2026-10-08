# Upgrading — Stem+MIDI Pro

How to roll out a new code version or new model weights on a small CPU deployment.

## Before you upgrade

1. Read [CHANGELOG.md](../../CHANGELOG.md) for the release and note breaking changes.
2. Back up `configs/model_config.yaml` and any checkpoint (`MODEL_CHECKPOINT_PATH`).
3. Uploads are ephemeral and there is no database — no data migration is needed.

## Code upgrade (bare-metal / venv)

```bash
cd /path/to/stem-midi-pro
git fetch && git checkout <release-tag>
source venv/bin/activate
pip install -r requirements.txt        # dependency changes
python demo.py                         # quick smoke test
# then restart the service
sudo systemctl restart stem-midi-pro   # or restart your uvicorn process
curl localhost:8000/health             # verify 200
curl localhost:8000/model-info         # verify expected config
```

## Model-weights upgrade

1. Place the new checkpoint beside the old one (do not overwrite in place).
2. Point `MODEL_CHECKPOINT_PATH` at the new file.
3. Restart, run one known test file, and compare `processing_report.json` metrics against the old baseline.
4. Keep the previous checkpoint until the new one has served production traffic for a few days.

## Model size / memory changes

If a release changes `d_model`/`n_layer`, re-check RAM headroom before deploying: process one 3-minute file while watching RSS. On 4–8 GB machines prefer the CPU-preset config (TODO CFG-9.1.3).

## Config compatibility

When upgrading, diff your local `configs/model_config.yaml` against the shipped one; new keys get defaults, removed keys may crash at startup (startup will fail fast with a clear error — check logs).
