# SPDX-License-Identifier: Apache-2.0
import logging

import torch
import torch.nn as nn

from models._mamba_compat import MAMBA_BACKEND, Mamba


def _dt_clamp_kwargs(section: dict) -> dict:
    """``dt_max`` is a fallback-only kwarg (the upstream kernel has no such arg)."""
    if MAMBA_BACKEND != "pure-torch":
        return {}
    return {"dt_max": float(section.get("dt_max", 0.1))}


logger = logging.getLogger(__name__)


# Inverse-STFT window size in frames. Bounds `torch.istft` scratch memory
# independently of input length (review §1: it allocates 4-5x its output).
_ISTFT_CHUNK_FRAMES = 512


class MambaSeparator(nn.Module):
    """
    Phase-coherent guitar/bass separation using Mamba SSM.

    Input:  audio (B, C=1, T) mono waveform
    Output: guitar_stem / bass_stem / residual (B, 1, T), None (no state
            cache yet - see C-2.5), metrics (B, 2) =
            [centroid_separation_proxy, phase_coherence]

    Layer structure is a pre-norm residual stack: every mixer is wrapped as
    ``h = h + mix(norm(h))`` with a final LayerNorm. Stacking bare mixers (the
    original behaviour) multiplied the activation RMS by ~0.02 per block, so a
    12-layer stack collapsed to ~1e-20 and every mask saturated at
    sigmoid(0) = 0.5 - i.e. the "separator" was a flat 0.5x gain. See C-1 in
    AUDIT_REPORT.md and tests/test_numerics.py.

    Masks are softmax-normalised across stems by default
    (``separator.mask_normalization: softmax``), so guitar + bass + residual
    reconstruct the mixture by construction; ``independent`` restores per-stem
    sigmoids.

    Streaming (C-2.5): ``forward(audio, state=...)`` threads the real SSM
    accumulator and the causal convolution's history between calls, which makes
    a chunk-contiguous stream numerically identical to one long call
    (``tests/test_streaming_state.py`` measures 4e-9). ``return_state=True``
    hands the next state back in the fourth return slot. The default file path
    still separates overlapping windows and crossfades them: with 50 % overlap
    there is no single recurrence to continue, so state threading there would be
    meaningless.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.hop_length = cfg["audio"]["hop_length"]
        self.n_fft = cfg["audio"]["n_fft"]
        d_model = cfg["separator"]["d_model"]

        n_layers = cfg["separator"].get("n_layers", cfg["separator"].get("n_layer"))
        if not n_layers or int(n_layers) <= 0:
            raise ValueError(
                "separator config must define a positive 'n_layers' (or legacy "
                f"'n_layer') key, got {n_layers!r}"
            )
        n_layers = int(n_layers)

        gradient_checkpointing = bool(
            (cfg.get("training") or {}).get("gradient_checkpointing", False)
        )
        if MAMBA_BACKEND == "pure-torch" and gradient_checkpointing:
            logger.info(
                "Separator: gradient checkpointing enabled for the pure-torch "
                "Mamba blocks (training memory bounded per block)."
            )

        # STFT encoder (fixed, no grad). persistent=False keeps these out of
        # state_dict, so checkpoint compatibility is unaffected.
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.register_buffer(
            "freqs",
            torch.linspace(0, self.n_fft // 2, self.n_fft // 2 + 1).view(1, -1, 1),
            persistent=False,
        )

        # Learnable projection: spectral bins → Mamba dimension
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 1, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

        # Mamba backbone - valid mamba-ssm kwargs only (no causal_conv1d_impl /
        # selective_scan_impl / use_fast_path; CPU uses the reference impl).
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=cfg["separator"]["d_state"],
                    d_conv=cfg["separator"]["d_conv"],
                    expand=cfg["separator"]["expand"],
                    gradient_checkpointing=gradient_checkpointing,
                    **_dt_clamp_kwargs(cfg["separator"]),
                )
                for _ in range(n_layers)
            ]
        )

        # Pre-norm partners for the residual stack (C-1) + final norm.
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # Mask normalisation: "softmax" (default) makes the three masks sum to 1
        # at every bin - mixture consistency, and an untrained model can no
        # longer emit stems whose total energy is ~0.26x the mix (C-1).
        self.mask_normalization = cfg["separator"].get("mask_normalization", "softmax")
        if self.mask_normalization not in ("softmax", "independent"):
            raise ValueError(
                "separator.mask_normalization must be 'softmax' or 'independent', "
                f"got {self.mask_normalization!r}"
            )

        # Multi-head output: guitar, bass, residual masks
        self.mask_heads = nn.ModuleDict(
            {
                "guitar": nn.Linear(d_model, self.n_fft // 2 + 1),
                "bass": nn.Linear(d_model, self.n_fft // 2 + 1),
                "residual": nn.Linear(d_model, self.n_fft // 2 + 1),
            }
        )

        # Phase coherence head (auxiliary loss)
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        logger.info("MambaSeparator built: backend=%s n_layers=%d", MAMBA_BACKEND, n_layers)

    def forward(self, audio: torch.Tensor, state: list | None = None, return_state: bool = False):
        if audio.dim() != 3:
            raise ValueError(
                f"MambaSeparator expects audio (B, C, T), got shape {tuple(audio.shape)}"
            )
        _B, C, T = audio.shape
        # NaN/Inf survives the STFT, the masks and the ISTFT, so a corrupt
        # upload becomes a delivered NaN stem (IO-3). Rejecting here also covers
        # callers that bypass audio_io (backends, tests, notebooks).
        if not torch.isfinite(audio).all():
            raise ValueError(
                "MambaSeparator received non-finite audio (NaN/Inf); the render would be "
                "non-finite too. Validate the input before separation."
            )
        if C != 1:
            raise ValueError(f"MambaSeparator expects mono input (C=1), got C={C}")
        if self.n_fft > T:
            raise ValueError(
                f"Input length {T} is shorter than n_fft={self.n_fft}; "
                "torch.stft cannot produce a frame. Pad or supply longer audio."
            )
        # IO-3 defense in depth: `audio_io` rejects non-finite uploads, but the
        # module is also used directly (training, tests, other callers), and a
        # single NaN sample propagates through STFT -> masks -> ISTFT to every
        # output sample. Refuse rather than return a NaN render that a PCM_16
        # encoder would turn into full-scale -1.0 DC.
        if not torch.isfinite(audio).all():
            raise ValueError(
                "MambaSeparator received non-finite input (NaN/Inf); refusing to "
                "separate it — every output sample would be non-finite."
            )

        # STFT: (B, C, T) → (B, F, T_frames)
        spec = torch.stft(
            audio.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        # Store the complex spectrogram in the (B, T_f, F) layout the mask heads
        # produce. Its inverse is non-contiguous, and multiplying a
        # non-contiguous complex tensor by a mask makes PyTorch materialise a
        # strided copy: +810 MB per stem at 600 s (review §1).
        spec = spec.transpose(1, 2).contiguous()  # (B, T_f, F)
        mag = torch.abs(spec)  # (B, T_f, F) - cached, reused by the metric below
        expected_bins = self.n_fft // 2 + 1
        if mag.shape[2] != expected_bins:
            raise ValueError(f"STFT freq bins {mag.shape[1]} != n_fft//2+1 ({expected_bins})")

        # Project to Mamba dim: (B, T_f, F) → (B, T_f, D)
        x = mag  # already (B, T_f, F)
        # Tonality of the INPUT (1 - Wiener entropy of the magnitude spectrum),
        # averaged over frames. Used to weight the separation score: a
        # centroid-ratio alone rewards broadband/distorted content, so a
        # hard-clipped square scored HIGHER than the clean sine it was made from
        # (GATE-4). Tonality is ~1 for a pure tone, lower for a square (odd
        # harmonics), ~0 for white noise and 0 for silence.
        tonality = self._tonality(x)
        hidden = self.input_proj(x)  # (B, T_f, D)
        del mag  # input_proj output kept; raw magnitude no longer needed
        if hidden.shape[-1] != self.mamba_blocks[0].d_model:
            raise ValueError(
                f"hidden dim {hidden.shape[-1]} != d_model {self.mamba_blocks[0].d_model}"
            )

        # Pre-norm residual Mamba stack (C-1). In streaming mode (C-2.5) each
        # block hands back its SSM accumulator + conv history, so the next call
        # continues the same recurrence instead of restarting it.
        next_state: list | None = [] if (return_state or state is not None) else None
        for i, (norm, block) in enumerate(zip(self.block_norms, self.mamba_blocks, strict=True)):
            if next_state is None:
                hidden = hidden + block(norm(hidden))
            else:
                block_state = None if state is None else state[i]
                mix, carried = block(norm(hidden), state=block_state, return_state=True)
                hidden = hidden + mix
                next_state.append(carried)
        hidden = self.final_norm(hidden)

        # Apply masks directly to the complex spectrogram: mask_mag * spec is
        # equivalent to (mask_mag * mag) * exp(i*phase). Reuses the cached STFT.
        # Process ONE stem at a time - compute its mask, apply, ISTFT, and drop
        # everything but the reduced centroid - so peak memory stays O(two
        # spectrograms) instead of O(three masks + three magnitudes + spec)
        # on the 4-8 GB CPU target.
        # Mask logits for all stems (needed together for a softmax across
        # stems). One (B, T_f, F) f32 tensor per stem: ~21 MB at a 60 s segment.
        logits = {name: head(hidden) for name, head in self.mask_heads.items()}
        if self.mask_normalization == "softmax":
            stacked = torch.stack(list(logits.values()), dim=0)  # (S, B, T_f, F)
            probs = torch.softmax(stacked, dim=0)
            masks = {name: probs[i] for i, name in enumerate(logits)}
            del stacked, probs
        else:
            masks = {name: torch.sigmoid(v) for name, v in logits.items()}
        del logits

        # Apply masks directly to the complex spectrogram: mask * spec is
        # equivalent to (mask * mag) * exp(i*phase), so the cached STFT is
        # reused. Each stem is reduced to audio (and, for guitar/bass, a
        # windowed centroid) and then dropped, so peak memory stays O(two
        # spectrograms) instead of O(three masks + three magnitudes + spec).
        stems = {}
        centroids = {}
        for name, mask in masks.items():
            # Only guitar and bass feed the centroid metric; the residual's
            # magnitude used to be computed and never read.
            if name in ("guitar", "bass"):
                centroids[name] = self._masked_centroid(mask, spec)
            stems[name] = self._masked_stem(mask, spec, length=T).unsqueeze(1)
        del masks, spec

        phase_score = self.phase_head(hidden.mean(dim=1))  # (B, 1)

        # Perceptual metric (spectral centroid separation proxy - NOT real SI-SDR).
        # No additional STFT (C-2.6 / PERF-7.1).
        centroid_metric = self._centroid_separation(centroids["guitar"], centroids["bass"])
        # Gate by the input's tonality so distortion cannot earn a better score
        # than the clean signal it came from (GATE-4). Cubed because the gate has
        # to out-run the centroid term: a hard-clipped square wave saturates that
        # term at its 30 dB clamp (measured 30.0 x tonality 0.64) while a clean
        # sine sits near 12 dB, so a linear weight still ranked the square higher.
        separation_metric = centroid_metric * tonality.pow(3)
        metrics = torch.stack([separation_metric, phase_score.squeeze(-1)], dim=-1)

        # ``new_state`` is None unless the caller asked for streaming: the
        # chunked file path overlaps 50% of the waveform, which makes a single
        # recurrence across chunks ill-defined (the crossfade handles it).
        return (
            stems["guitar"],
            stems["bass"],
            stems["residual"],
            next_state,
            metrics,
        )

    def _frame_windows(self, n_frames: int):
        """Yield (start, stop) frame windows used by every chunked stage."""
        step = max(1, _ISTFT_CHUNK_FRAMES)
        for start in range(0, n_frames, step):
            yield start, min(start + step, n_frames)

    def _mask_window(self, mask_tf: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Contiguous (B, c, F) slice of a precomputed (B, T_f, F) mask.

        Slicing the *mask output* rather than `hidden[:, start:stop]` matters:
        slicing hidden produces a copy of the mamba output (~300 MB at 600 s),
        while slicing the mask yields a 4 MB view per window (review §1).
        """
        return mask_tf[:, start:stop]

    def _masked_centroid(self, mask_tf: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """Spectral centroid of the masked stem, (B, T_f), computed windowed.

        The old path materialised a full ``(B, T_f, F)`` masked-magnitude
        spectrogram per stem purely to feed this metric — 404 MB each at
        600 s, and the residual stem's copy was computed and never read
        (review §1). Magnitudes are now accumulated window by window; the
        centroid is identical because it is a per-frame reduction.
        """
        freqs = torch.linspace(
            0, self.n_fft // 2, self.n_fft // 2 + 1, device=spec.device, dtype=spec.real.dtype
        ).view(1, -1)
        out = torch.empty(spec.shape[0], spec.shape[1], device=spec.device, dtype=spec.real.dtype)
        for start, stop in self._frame_windows(spec.shape[1]):
            mag = torch.abs(mask_tf[:, start:stop] * spec[:, start:stop])  # (B, c, F)
            out[:, start:stop] = (mag * freqs).sum(dim=-1) / (mag.sum(dim=-1) + 1e-8)
        return out

    def _masked_stem(self, mask_tf: torch.Tensor, spec: torch.Tensor, length: int) -> torch.Tensor:
        """istft of one masked stem; returns (B, T).

        The masked spectrogram is assembled window by window (so no full-size
        mask copy is ever live) and then handed to ``_istft_chunked``, which
        performs the overlap-add reconstruction that is verified sample-exact
        against a whole-file ``torch.istft``. Reconstructing each mask window
        independently would produce hard seams at the window boundaries — the
        per-window and whole-file reconstructions differ by ~2e-2 otherwise.
        """
        masked = torch.empty_like(spec)  # (B, T_f, F) complex
        for start, stop in self._frame_windows(spec.shape[1]):
            masked[:, start:stop] = mask_tf[:, start:stop] * spec[:, start:stop]
        out = self._istft_chunked(masked.transpose(1, 2), length=length)
        del masked
        return out

    def _istft_chunked(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Inverse STFT in frame windows with overlap-add; returns (B, length).

        ``torch.istft`` over a whole multi-minute spectrogram allocates ~4-5x
        its own output size internally (measured: 1216 MB of scratch for a
        264 MB 600 s output), which dominated peak RSS for long inputs. Each
        window here is only ``_ISTFT_CHUNK_FRAMES`` frames, so the scratch is
        a few MB regardless of input length.

        Each window is reconstructed with ``pad`` extra frames on either side
        and only its interior is copied out. That padding is what makes the
        result exact: ``torch.istft`` normalises by the window sum, and samples
        within ``n_fft`` of a window edge see contributions only from the
        frames present in that window. Copying the interior (>= n_fft // 2
        samples in) means every copied sample carries the same complete window
        sum it has in the whole-file transform. Skipping the padding is a
        silent quality bug: the seams were ~3e-3 peak error, audible on
        transient material (review §1).
        """
        n_frames = spec.shape[-1]
        pad = (self.n_fft // 2 + self.hop_length - 1) // self.hop_length
        step = max(1, _ISTFT_CHUNK_FRAMES)
        out = torch.zeros(spec.shape[0], length, device=spec.device, dtype=spec.real.dtype)
        for start in range(0, n_frames, step):
            stop = min(start + step, n_frames)
            # Outer boundaries need no padding: frame 0 / frame N-1 see the
            # same truncated window sum as they do in the whole-file transform.
            lo_frame = max(0, start - pad)
            hi_frame = min(n_frames, stop + pad)
            audio = torch.istft(
                spec[..., lo_frame:hi_frame],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                length=(hi_frame - lo_frame) * self.hop_length,
            )
            # Chunk sample u corresponds to global sample lo_frame*hop + u.
            out_lo = lo_frame * self.hop_length
            take_lo = start * self.hop_length - out_lo
            take_hi = min(stop * self.hop_length, length) - out_lo
            take_hi = min(take_hi, audio.shape[-1])
            if take_hi > take_lo:
                out[:, out_lo + take_lo : out_lo + take_hi] = audio[:, take_lo:take_hi]
        return out

    def _tonality(self, mag: torch.Tensor) -> torch.Tensor:
        """Per-batch tonality in [0, 1]: ``1 -`` Wiener entropy of the spectrum.

        Wiener entropy is the geometric mean over the arithmetic mean of the
        magnitude spectrum: ~0 for a pure tone (all energy in one bin), ~1 for
        flat noise. Silence has no spectrum to speak of and returns 0, so a
        zero-energy input cannot earn any separation score.
        """
        power = mag.clamp(min=0) + 1e-12
        log_mean = torch.log(power).mean(dim=-1)  # (B, T_f)
        arith_mean = power.mean(dim=-1)
        flatness = torch.exp(log_mean) / arith_mean.clamp(min=1e-12)  # (B, T_f)
        energy = mag.pow(2).sum(dim=-1)  # (B, T_f)
        tonality = (1.0 - flatness).clamp(0.0, 1.0)
        # Frames with no energy carry no tonality information.
        tonality = tonality * (energy > 1e-12).to(tonality.dtype)
        return tonality.mean(dim=-1)  # (B,)

    def _centroid_separation(
        self, guitar_centroid: torch.Tensor, bass_centroid: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral centroid separation heuristic - NOT SI-SDR.

        Takes per-frame centroid vectors ``(B, T_f)`` reduced inside
        ``forward`` (no additional STFT; no full magnitude retention —
        C-2.6 / PERF-7.1 / C-1). "Are the stems spectrally distinct?"
        True SI-SDR requires a ground-truth reference and is computed
        only during training (via PerceptualAudioLoss).
        Renamed from _estimate_si_sdr per C-2.6 / TODO.md.
        """
        separation = torch.abs(guitar_centroid - bass_centroid).mean(dim=-1)
        return torch.clamp(10 * torch.log10(separation + 1e-8) + 20, min=0, max=30)

    def _spectral_centroid(self, mag: torch.Tensor) -> torch.Tensor:
        """Spectral centroid per frame from a magnitude spectrogram (B, F, T_f)."""
        freqs = self.freqs.to(dtype=mag.dtype)
        centroid = (mag * freqs).sum(dim=1) / (mag.sum(dim=1) + 1e-8)
        return centroid  # (B, T_frames)
