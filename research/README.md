# Research code

<!-- STATUS: research -->

This directory is experimental and is not the runtime implementation for the
root FastAPI/model path.

## Contents

- `mamba3_per_track/` — separate per-track Mamba-3 experiment, with its own
  dataset and training loop.
- `mamba-ssm-reference/mamba/` — tracked snapshot of the upstream
  [state-spaces/mamba](https://github.com/state-spaces/mamba) repository. It is
  reference material, not a runtime dependency. The snapshot does not contain
  an embedded `.git` directory or act as a Git submodule.

An identical Mamba snapshot is also present under `stem_midi_pro/mamba/` in the
legacy duplicate tree. That duplication is unresolved and should be reviewed
before deleting either copy.

## Mamba-3 training CLI

`research/mamba3_per_track/train.py` takes command-line flags; it does not read
`config.yaml` and does not accept `--config`. Its default device is `cuda`, not
CPU. A short CPU experiment can be requested with flags such as:

```bash
python research/mamba3_per_track/train.py \
  --data /path/to/musdb18hq/train \
  --save ./outputs/mamba3-research \
  --steps 50 \
  --batch 2 \
  --device cpu \
  --no-amp
```

The CLI options are visible in the source. Execution is not verified in this
environment, and the model imports `mamba_ssm`; do not assume the CPU command
works with every Mamba-SSM build. See [the research setup note](mamba3_per_track/SETUP.md).

## Test and syntax discovery

- `pytest.ini` limits ordinary `pytest` discovery to `tests/`, excluding
  research and upstream vendor tests unless explicitly requested.
- `check_syntax.py` skips both vendored Mamba source trees.
- `.github/workflows/codeql.yml` is the only checked-in workflow. There is no
  `ci.yml` lint/test workflow.
