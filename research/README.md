# /research — Mamba-3 experimental path

This directory is **research-only** and is not part of the production
v1 pipeline (which lives at the repo root: `main.py`, `api.py`, `models/*`,
`train.py`, `data/datasets.py`).

## Layout

```
research/
├── README.md                       (this file)
├── mamba3_per_track/               Mamba-3 per-track adaptive filter bank
│   ├── per_track_processor.py      (was: model.py)
│   ├── train.py                    (was: train_mamba3.py)
│   ├── losses.py                   (was: utils/losses_mamba3.py)
│   ├── mamba3_datasets.py          StemDataset, Slakh2100StemDataset, loaders
│   ├── config.yaml                 (was: configs/mamba3_config.yaml)
│   ├── SETUP.md                    (was: SETUP.md)
│   └── test_losses.py              (was: tests/test_losses_mamba3.py)
└── mamba-ssm-reference/            Vendored reference clone (see below)
    └── mamba/                      state-spaces/mamba @ vendored commit
        ├── .git/                   (DO NOT commit; own git history)
        ├── mamba_ssm/              Reference source
        ├── csrc/                   Reference CUDA kernels
        └── ...
```

## Why is this here?

The Mamba-3 work was started in parallel to the v1 production path but
uses a different architecture (per-track processor, CREPE-conditioned),
different data format (per-stem .wav + .npy pre-features), and a different
training loop. Rather than maintain two production codebases, the
production path is **v1** (see `ARCHITECTURE.md`), and Mamba-3 is
preserved here as a research reference for future work.

## Running the Mamba-3 research path

```bash
# CPU-only (default; no CUDA required for synthesis / dry-run)
python research/mamba3_per_track/train.py --config research/mamba3_per_track/config.yaml

# GPU (experimental; requires CUDA build of mamba_ssm + causal_conv1d)
python research/mamba3_per_track/train.py --config research/mamba3_per_track/config.yaml --device cuda
```

See `research/mamba3_per_track/SETUP.md` for the full setup including
the CUDA build instructions.

## Vendored mamba-ssm reference

`research/mamba-ssm-reference/mamba/` is a **vendored snapshot** of
[state-spaces/mamba](https://github.com/state-spaces/mamba) (the
official mamba-ssm reference implementation). It is included for:

- **Reference** — when implementing or debugging Mamba-style SSM layers.
- **Spec** — the Triton/Cute/TileLang kernels in `mamba/mamba_ssm/ops/*mamba3*`
  are the upstream source-of-truth for what Mamba-3 should look like.

It is **not a runtime dependency**. The v1 production path does not
import from this directory, and `requirements.txt` installs `mamba-ssm`
from PyPI.

The vendored clone carries its own `.git/` directory and is not
intended to be a submodule or to receive upstream pulls. See
`research/mamba-ssm-reference/mamba/.gitignore` for the local excludes.

## CI / lint exclusion

- `check_syntax.py` skips `research/mamba-ssm-reference/`.
- `.github/workflows/ci.yml` excludes it from flake8/black/isort.
- `pytest` does not collect anything under `research/` (no `__init__.py`).

## Re-syncing with upstream

If you need a fresh snapshot of state-spaces/mamba:

```bash
rm -rf research/mamba-ssm-reference/mamba
git clone --depth 1 https://github.com/state-spaces/mamba.git research/mamba-ssm-reference/mamba
```
