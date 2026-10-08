# ADR 0002: CPU-Only Target (Bare-Metal i5)

**Status:** Accepted (TODO D-STRAT-3, DEP-4.1.6, Section 7)
**Date:** 2026-06

## Context

The original design assumed NVIDIA H100/A100 GPUs with TensorRT-LLM, FP8, CUDA graphs, and `causal-conv1d`. The actual deployment target for this product is a bare-metal Intel i5 with 4–8 GB RAM, for indie studios and self-hosters without GPUs.

## Decision

Drop all CUDA-only paths and dependencies: no `causal-conv1d`, no TensorRT-LLM export, no FP8. `mamba-ssm` runs via its pure-PyTorch reference implementation on CPU. Downsize model configs to fit 4–8 GB RAM. Accept the honest performance envelope: **~30–60 s of processing per minute of audio, one request at a time**.

## Consequences

- Radically simpler deployment (no drivers, no GPU provisioning) and smaller dependency tree.
- Streaming/chunked paths must be validated on CPU memory limits (TODO C-2.5, API-6.13).
- GPU-enthusiast latency claims in old docs were removed or tagged `[REMOVED]` rather than kept as goals.
- If GPU support returns later, it is an optional accel path, never a requirement.
