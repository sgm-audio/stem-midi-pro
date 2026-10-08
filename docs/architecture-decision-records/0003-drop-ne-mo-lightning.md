# ADR 0003: Drop NeMo and PyTorch Lightning

**Status:** Accepted (TODO RF-3.1.7–9, TR-12.3, DEP-4.1.3/4.1.4)
**Date:** 2026-06

## Context

`main.py` wraps the model in NeMo `ModelPT` with neural-type decorators, and `train.py` uses PyTorch Lightning. On the CPU-only bare-metal target, NeMo (`nemo-toolkit[all]`) pulls ~500 MB of CUDA-leaning dependencies, and Lightning's abstractions add indirection without buying anything for a single-process CPU loop.

## Decision

Replace the NeMo `ModelPT` wrapper with a plain `torch.nn.Module` (`StemMidiModel`) exposing `forward`, `training_step`, `validation_step`. Replace Lightning's `Trainer` with a plain PyTorch training loop (custom loop with gradient accumulation, bf16 autocast on CPU, checkpointing, early stopping). Remove both packages from `requirements.txt`.

## Consequences

- Much smaller install footprint; no CUDA-adjacent transitive deps on a CPU machine.
- Explicit control over training semantics (e.g. determinism flags, `ckpt_path='best'` magic goes away — TODO TR-12.2).
- We re-implement some niceties (logging, checkpointing) ourselves; acceptable at this scale.
- NeMo typing decorators are replaced by explicit shape asserts in `forward` (RF-3.1.8).
