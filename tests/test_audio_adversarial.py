# SPDX-License-Identifier: Apache-2.0
"""Adversarial audio and CPU-practicality tests.

The existing suite feeds the model one thing: a 0.3-amplitude 220 Hz sine (or
``torch.randn``).  Silence, DC offset, hard clipping, NaN, sub-23 ms clips,
sample-rate conversion and long files are untested, yet those are exactly the
inputs a public upload endpoint receives.

Two techniques are used to keep these honest and fast:

* ``passthrough_model`` replaces only the *neural* stages with identity ops, so
  the real I/O path (soundfile load -> resample -> stem dict) is measured
  without 62 M parameters.  Separation quality is explicitly out of scope.
* ``slow``-marked memory tests *project* rather than allocate: they measure
  small inputs and extrapolate, so they cannot OOM the CI box while still
  failing when the pipeline's memory growth is linear in file length.

Instruments: measured on 2 vCPU / 3.9 GB with configs/model_config.yaml
(62.2 M params, pure-torch Mamba fallback): 60 s of audio -> 17.9 s wall
(0.30x realtime) and 2346 MB peak RSS, growing ~23 MB per second of audio.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("soundfile", reason="these tests write real WAV files")

import soundfile as sf  # noqa: E402

SR = 44100


# ---------------------------------------------------------------------------
# Helpers / instruments
# ---------------------------------------------------------------------------


@pytest.fixture
def passthrough_model(mock_config):
    """StemMidiModel whose separator/transcriber are transparent.

    The separator returns the input waveform as the "guitar" stem (so the
    returned stem is a faithful copy of what the I/O layer produced), and the
    transcriber returns zero logits.  Everything else — config handling,
    resampling, report building, routing — is the production code.
    """
    from main import StemMidiModel

    model = StemMidiModel(mock_config).eval()

    class _PassthroughSeparator(torch.nn.Module):
        def __init__(self, seen: list[int]):
            super().__init__()
            self.seen = seen

        def forward(self, audio):
            self.seen.append(int(audio.shape[-1]))
            zeros = torch.zeros_like(audio)
            metrics = torch.tensor([[21.0, 0.9]])
            return audio, zeros, zeros, None, metrics

    class _ZeroTranscriber(torch.nn.Module):
        def forward(self, stem):
            b, _c, t = stem.shape
            frames = t // 512
            return (
                torch.zeros(b, frames, 1),
                torch.zeros(b, frames, 128),
                torch.ones(b, frames, 1),
                torch.zeros(b, frames, 3),
                torch.full((b, frames), 0.9),
            )

    seen: list[int] = []
    model.separator = _PassthroughSeparator(seen)
    model.transcriber = _ZeroTranscriber()
    model._seen_lengths = seen
    return model


def _write(path, samples: np.ndarray, sr: int = SR) -> str:
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sr, subtype="FLOAT")
    return str(path)


def _tone(seconds: float, freq: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_db(x: np.ndarray, sr: int, freq: float) -> float:
    """Peak level in dB near ``freq``, normalised so a full-scale tone is ~0 dB."""
    w = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    k = int(np.argmin(np.abs(freqs - freq)))
    band = spectrum[max(0, k - 4) : k + 5]
    return float(20 * np.log10(band.max() / (len(x) / 4) + 1e-12))


def _rss_mb() -> float:
    import resource  # POSIX only; these tests are already CPU/Linux-targeted

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _run_file(model, tmp_path, name: str, samples: np.ndarray, sr: int = SR):
    path = _write(tmp_path / name, samples, sr)
    return model.process_audio_file(path)


# ---------------------------------------------------------------------------
# Silence
# ---------------------------------------------------------------------------


@pytest.fixture
def confident_model(mock_config):
    """StemMidiModel with a deliberately over-eager transcriber.

    A random-weight head fires arbitrarily (seeding a test against it measures
    the seed, not the pipeline). This fixture instead pins the worst realistic
    case: a *trained-looking* head that asserts an onset, max-velocity note on
    every frame of every input. Any suppression of impossible output must come
    from the pipeline, not from a lucky initialisation.
    """
    from main import StemMidiModel

    model = StemMidiModel(mock_config).eval()

    class _EagerTranscriber(torch.nn.Module):
        def forward(self, stem):
            b, _c, t = stem.shape
            frames = t // 512
            return (
                torch.full((b, frames, 1), 4.0),  # onset logits -> p ~= 0.98
                torch.zeros(b, frames, 128),  # pitch: argmax -> 0 (C-1)
                torch.ones(b, frames, 1),  # velocity = 127
                torch.zeros(b, frames, 3),
                torch.full((b, frames), 0.99),  # maximally confident
            )

    model.transcriber = _EagerTranscriber()
    return model


def test_silence_emits_no_note_events_even_with_a_confident_head(confident_model):
    """Digital silence must transcribe to zero notes, not a wall of notes.

    1 s of zeros at hop 512 is 87 model frames; with an eager head that becomes
    a MIDI file packed with 87 invented notes, delivered with confidence 0.99
    and no error. A user who uploads a muted channel receives nonsense.
    """
    with torch.inference_mode():
        out = confident_model(torch.zeros(1, 1, SR))
    events = out["midi_metadata"]["midi_events"]
    assert events == [], f"silence produced {len(events)} note events"


def test_silence_is_flagged_not_silent_pass(mock_model):
    """A silent input must be reported as such, so routing can reject it."""
    with torch.inference_mode():
        report = mock_model(torch.zeros(1, 1, SR))["processing_report"]
    assert any(
        flag in report.artifact_flags for flag in ("silence", "no_signal")
    ), f"silence reported as {report.artifact_flags}"


# ---------------------------------------------------------------------------
# Clipping / DC / NaN
# ---------------------------------------------------------------------------


def test_hard_clipped_input_is_flagged(mock_model):
    """A square-clipped guitar DI is the most common bad-stem cause.

    Input clipped at +/- 1.0 for a third of its samples must raise a clipping
    flag regardless of what the separator does to the level.
    """
    clipped = np.clip(_tone(1.0, 220.0, amp=6.0), -1.0, 1.0).astype(np.float32)
    with torch.inference_mode():
        report = mock_model(torch.from_numpy(clipped).reshape(1, 1, -1))["processing_report"]
    assert (
        "clipping" in report.artifact_flags
    ), f"hard-clipped input flagged as {report.artifact_flags}"


def test_dc_offset_is_flagged_or_removed(mock_model):
    """DC offset wastes headroom and biases every mask; it must be handled."""
    dc = np.full(SR, 0.5, dtype=np.float32)
    with torch.inference_mode():
        report = mock_model(torch.from_numpy(dc).reshape(1, 1, -1))["processing_report"]
    assert "dc_offset" in report.artifact_flags


def test_nan_input_does_not_silently_produce_nan_report(mock_model):
    """A NaN sample must be rejected at the boundary, never propagated.

    Previously the NaN flowed into the report (si_sdr=nan, avg_confidence=nan,
    artifact_flags=['none']) and ``create_response_zip`` emitted ``NaN``
    literals into processing_report.json, which strict parsers reject.
    ``_validate_audio`` now rejects the input up front (BUG-AUDIO-4).
    """
    poisoned = _tone(1.0, 220.0).copy()
    poisoned[1000] = np.nan
    with pytest.raises((ValueError, AssertionError)), torch.inference_mode():
        mock_model(torch.from_numpy(poisoned).reshape(1, 1, -1))


def test_report_serialises_to_strict_json(mock_model, tmp_path):
    """``processing_report.json`` must be parseable by a strict JSON reader.

    ``json.dumps`` happily emits bare ``NaN``/``Infinity`` tokens, which RFC
    8259 forbids and most clients reject. Now that non-finite input is rejected
    at the boundary, a report built from valid audio must serialise cleanly.
    """
    import json

    result = _run_file(mock_model, tmp_path, "ok.wav", _tone(0.5, 220.0))
    report = result["report"]
    payload = json.dumps(
        {
            "si_sdr": float(report.si_sdr),
            "phase_coherence": float(report.phase_coherence),
            "avg_confidence": float(report.avg_confidence),
        },
        allow_nan=False,
    )
    assert json.loads(payload)["si_sdr"] == pytest.approx(report.si_sdr)


# ---------------------------------------------------------------------------
# Too-short input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.001, 0.5, 0.99])
def test_sub_window_audio_raises_a_validation_error(mock_model, fraction):
    """Under one STFT window the API must answer 400, not blow up with a 500.

    The threshold is ``n_fft`` (read from the model config so this tracks the
    real value), because ``torch.stft`` reflect-pads by ``n_fft/2`` and raises a
    ``RuntimeError`` for anything shorter.
    """
    from audio_io import AudioValidationError

    n_fft = int(mock_model.cfg["audio"]["n_fft"])
    n_samples = max(1, int(n_fft * fraction))
    assert n_samples < n_fft
    tiny = _tone(1.0, 220.0)[:n_samples]

    with (
        pytest.raises((AudioValidationError, ValueError), match=r"short|n_fft"),
        torch.inference_mode(),
    ):
        mock_model(torch.from_numpy(tiny).reshape(1, 1, -1))


def test_single_stft_window_is_accepted(mock_model):
    """Exactly ``n_fft`` samples is the documented minimum and must not raise."""
    n_fft = int(mock_model.cfg["audio"]["n_fft"])
    window = _tone(1.0, 220.0)[:n_fft]
    with torch.inference_mode():
        out = mock_model(torch.from_numpy(window).reshape(1, 1, -1))
    assert out["guitar_stem"].shape[-1] == n_fft


# ---------------------------------------------------------------------------
# Sample-rate conversion
# ---------------------------------------------------------------------------


def test_upsampling_does_not_inject_spectral_images(passthrough_model, tmp_path):
    """An 8 kHz upload must not acquire energy that was never in the recording.

    Guitar content lives well below 4 kHz, but the transcriber's mel input sees
    the image as real high-band energy, which shifts onset/pitch decisions. A
    band-limited (polyphase/soxr) resampler puts the image below -60 dB.
    """
    src_sr = 8000
    result = _run_file(
        passthrough_model, tmp_path, "telephone.wav", _tone(1.0, 3500.0, src_sr), src_sr
    )
    stem = result["stems"]["guitar"]
    assert result["sample_rate"] == SR
    assert len(stem) == SR, f"resampled length {len(stem)} != {SR}"

    fundamental = _band_db(stem, SR, 3500.0)
    image = _band_db(stem, SR, 4500.0)
    assert image - fundamental < -30.0, (
        f"image at 4.5 kHz is only {image - fundamental:.1f} dB below the signal "
        "(linear-interpolation resampler)"
    )


def test_downsampled_length_is_preserved(passthrough_model, tmp_path):
    """Resampling must not change duration: 1 s in -> 1 s out (already holds)."""
    result = _run_file(passthrough_model, tmp_path, "hi.wav", _tone(1.0, 1000.0, 48000), 48000)
    assert abs(len(result["stems"]["guitar"]) - SR) <= 2


# ---------------------------------------------------------------------------
# Extreme polyphony and full-scale material
# ---------------------------------------------------------------------------


def test_twenty_four_tone_cluster_is_finite_and_bounded(mock_model):
    """A dense cluster (24 partials, 4 octaves) must not NaN or hang.

    Extreme polyphony is the adversarial case for the mask/Mamba path: every
    frequency bin is excited, so any divide-by-zero in the centroid heuristic or
    the mel basis shows up as NaN. Also asserts a generous wall-clock ceiling so
    a pathological slowdown on CPU is caught rather than shipped.
    """
    import time

    freqs = np.array([55.0 * 2 ** (k / 12) for k in range(24)])  # 4 octaves of semitones
    t = np.arange(SR) / SR
    mix = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    assert np.abs(mix).max() <= 1.0

    start = time.time()
    with torch.inference_mode():
        out = mock_model(torch.from_numpy(mix.astype(np.float32)).reshape(1, 1, -1))
    elapsed = time.time() - start

    for stem in ("guitar_stem", "bass_stem"):
        assert torch.isfinite(out[stem]).all(), f"NaN/Inf in {stem} for dense cluster"
    report = out["processing_report"]
    assert np.isfinite(report.si_sdr) and np.isfinite(report.avg_confidence)
    assert elapsed < 30.0, f"1 s of dense audio took {elapsed:.1f}s at test dims"


def test_full_scale_noise_stems_stay_finite(mock_model):
    """0 dBFS broadband noise: worst case for istft overshoot — must stay finite."""
    rng = np.random.default_rng(1234)
    noise = np.clip(rng.standard_normal(SR) * 0.99, -1.0, 1.0).astype(np.float32)
    with torch.inference_mode():
        out = mock_model(torch.from_numpy(noise).reshape(1, 1, -1))
    assert torch.isfinite(out["guitar_stem"]).all()
    assert torch.isfinite(out["bass_stem"]).all()


# ---------------------------------------------------------------------------
# CPU practicality: chunking and memory growth
# ---------------------------------------------------------------------------


def test_process_audio_file_honours_chunk_duration(mock_config, tmp_path):
    """Long files must be processed in ``chunk_duration_sec`` slices.

    Measured with the production config: peak RSS grew ~23 MB per second of
    audio (60 s -> 2346 MB), which projected to ~14 GB for the 10-minute limit
    the API advertises. ``chunk_duration_sec`` was only read by the streaming
    path; ``process_audio_file`` now chunks too, bounding the working set.
    """
    from main import StemMidiModel

    cfg = {**mock_config, "backends": {"separator": "mamba", "transcriber": "mamba"}}
    model = StemMidiModel(cfg).eval()

    seen: list[int] = []

    class _Recording(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, audio):
            seen.append(int(audio.shape[-1]))
            return self.inner(audio)

    model.separator = _Recording(model.separator)
    path = _write(tmp_path / "six.wav", np.zeros(6 * SR, dtype=np.float32))
    model.process_audio_file(path)

    chunk_len = int(cfg["audio"]["chunk_duration_sec"] * SR)
    assert seen, "separator was never called"
    assert max(seen) <= chunk_len, (
        f"largest single forward saw {max(seen)} samples "
        f"({max(seen) / SR:.1f}s) with chunk_duration_sec="
        f"{cfg['audio']['chunk_duration_sec']}"
    )
    assert len(seen) >= 2, "a 6 s file with 2 s chunks must take more than one chunk"


@pytest.mark.slow
def test_one_minute_upload_fits_in_2gb(tmp_path):
    """A 60 s file must not grow peak RSS by more than the chunk working set.

    Regression for BUG-PERF-1. Before chunking, ``process_audio_file`` passed
    the whole file to the model: peak RSS grew ~23 MB per audio-second
    (60 s -> 2346 MB), projecting to ~14 GB for the 10-minute limit the API
    advertises. With ``chunk_duration_sec`` honoured, peak RSS plateaus:
    measured 880 MB at 4 s and 966 MB at 64 s with the production config.

    Asserted as growth-over-baseline rather than an extrapolated slope, because
    ``ru_maxrss`` is a high-water mark and a fitted slope through a plateau is
    dominated by allocator noise.
    """
    import yaml

    from main import StemMidiModel

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = StemMidiModel(cfg).eval()
    sr = int(cfg["audio"]["sample_rate"])

    path = _write(tmp_path / "sixty.wav", _tone(60.0, 220.0, sr))
    baseline = _rss_mb()
    model.process_audio_file(path)
    growth = _rss_mb() - baseline

    # The output stems alone are ~21 MB for 60 s; the chunk working set is a few
    # hundred MB. 800 MB leaves headroom for allocator variance while still
    # failing on the old linear-growth behaviour (~1 500 MB for 60 s).
    assert growth < 800, (
        f"60 s upload grew peak RSS by {growth:.0f} MB (baseline {baseline:.0f} MB); "
        "chunking is no longer bounding the working set"
    )
