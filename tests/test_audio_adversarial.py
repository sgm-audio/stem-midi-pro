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


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-1: nothing gates the transcriber on input energy, so a "
    "silent file is transcribed at face value (one note per model frame)",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-2: silence yields artifact_flags=['none'] — the gate has no "
    "notion of 'no signal'",
)
def test_silence_is_flagged_not_silent_pass(mock_model):
    """A silent input must be reported as such, so routing can reject it."""
    with torch.inference_mode():
        report = mock_model(torch.zeros(1, 1, SR))["processing_report"]
    assert any(flag in report.artifact_flags for flag in ("silence", "no_signal")), (
        f"silence reported as {report.artifact_flags}"
    )


# ---------------------------------------------------------------------------
# Clipping / DC / NaN
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-3: clipping is tested on model *output* only "
    "(abs().max() > 0.99). A hard-clipped input shrinks under the masks, so it "
    "comes back labelled 'noise_floor' — a different defect than the one present",
)
def test_hard_clipped_input_is_flagged(mock_model):
    """A square-clipped guitar DI is the most common bad-stem cause.

    Input clipped at +/- 1.0 for a third of its samples must raise a clipping
    flag regardless of what the separator does to the level.
    """
    clipped = np.clip(_tone(1.0, 220.0, amp=6.0), -1.0, 1.0).astype(np.float32)
    with torch.inference_mode():
        report = mock_model(torch.from_numpy(clipped).reshape(1, 1, -1))["processing_report"]
    assert "clipping" in report.artifact_flags, (
        f"hard-clipped input flagged as {report.artifact_flags}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-4: a constant DC offset is passed through unflagged "
    "(centroid proxy collapses to si_sdr=0 but no artifact is raised)",
)
def test_dc_offset_is_flagged_or_removed(mock_model):
    """DC offset wastes headroom and biases every mask; it must be handled."""
    dc = np.full(SR, 0.5, dtype=np.float32)
    with torch.inference_mode():
        report = mock_model(torch.from_numpy(dc).reshape(1, 1, -1))["processing_report"]
    assert "dc_offset" in report.artifact_flags


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-4a: a single NaN sample is not rejected at the boundary and "
    "propagates through STFT/mel to the processing report",
)
def test_nan_input_does_not_silently_produce_nan_report(mock_model):
    """A NaN sample must be rejected at the boundary, never propagated.

    Today the NaN flows into the report: si_sdr=nan, avg_confidence=nan,
    artifact_flags=['none'], and ``create_response_zip`` then emits ``NaN``
    literals in processing_report.json — which strict JSON parsers reject.
    """
    poisoned = _tone(1.0, 220.0).copy()
    poisoned[1000] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        with torch.inference_mode():
            mock_model(torch.from_numpy(poisoned).reshape(1, 1, -1))


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-4b: NaN propagates into ProcessingReport (si_sdr=nan), so "
    "processing_report.json is not strict JSON",
)
def test_report_is_strict_json_for_nan_input(mock_model):
    """Whatever survives inference must serialise to strict JSON."""
    import json

    poisoned = _tone(0.5, 220.0).copy()
    poisoned[100] = np.nan
    with torch.inference_mode():
        out = mock_model(torch.from_numpy(poisoned).reshape(1, 1, -1))
    report = out["processing_report"]
    payload = json.dumps(
        {"si_sdr": float(report.si_sdr), "avg_confidence": float(report.avg_confidence)},
        allow_nan=False,
    )
    json.loads(payload)


# ---------------------------------------------------------------------------
# Too-short input
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-5: input shorter than n_fft/2 (1024 samples, 23 ms at the "
    "production config) raises RuntimeError from torch.stft reflect padding -> "
    "the API maps it to 500 'Processing failed' instead of 400",
)
@pytest.mark.parametrize("n_samples", [1, 200, 1000])
def test_sub_window_audio_raises_a_validation_error(mock_model, n_samples):
    """Under one STFT window the API must answer 400, not blow up with a 500."""
    from audio_io import AudioValidationError

    tiny = _tone(1.0, 220.0)[:n_samples]
    with pytest.raises((AudioValidationError, ValueError)):
        with torch.inference_mode():
            mock_model(torch.from_numpy(tiny).reshape(1, 1, -1))


# ---------------------------------------------------------------------------
# Sample-rate conversion
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG-AUDIO-6: resampling is F.interpolate(mode='linear'), a triangle "
    "filter with no anti-imaging; an 8 kHz source keeps a -4 dB image at 4.5 kHz",
)
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
    result = _run_file(
        passthrough_model, tmp_path, "hi.wav", _tone(1.0, 1000.0, 48000), 48000
    )
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


@pytest.mark.xfail(
    strict=True,
    reason="BUG-PERF-1: process_audio_file passes the whole file to the model; "
    "cfg['audio']['chunk_duration_sec'] is only read by process_audio_streaming, "
    "so a 10-min upload is one 51k-frame STFT (~14 GB projected peak RSS)",
)
def test_process_audio_file_honours_chunk_duration(mock_config, tmp_path):
    """Long files must be processed in ``chunk_duration_sec`` slices.

    Measured with the production config: peak RSS grows ~23 MB per second of
    audio (60 s -> 2346 MB), which projects to ~14 GB for the 10-minute limit
    the API advertises. Chunking is the configured remedy and is currently
    bypassed on the non-streaming path.
    """
    from main import StemMidiModel

    model = StemMidiModel(mock_config).eval()
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

    chunk_len = int(mock_config["audio"]["chunk_duration_sec"] * SR)
    assert max(seen) <= chunk_len, (
        f"largest single forward saw {max(seen)} samples "
        f"({max(seen) / SR:.1f}s) with chunk_duration_sec="
        f"{mock_config['audio']['chunk_duration_sec']}"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="BUG-PERF-1: peak RSS is linear in file length (~23 MB/s), so the "
    "10-minute API limit needs ~14 GB — far beyond the 4-8 GB target",
)
def test_ten_minute_upload_fits_in_4gb():
    """Project 10-minute memory from a cheap 12 s measurement.

    Measuring beats asserting: this uses the production config and extrapolates
    from the observed slope instead of allocating the 10-minute tensor (which
    would OOM the test runner itself). The 4 GB ceiling is the stated CPU
    target's low end.
    """
    import yaml

    from main import StemMidiModel

    with open("configs/model_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = StemMidiModel(cfg).eval()

    baseline = _rss_mb()
    peaks = []
    for seconds in (4, 12):
        with torch.inference_mode():
            model(torch.randn(1, 1, seconds * SR) * 0.05)
        peaks.append((seconds, _rss_mb()))

    (s1, m1), (s2, m2) = peaks
    mb_per_second = (m2 - m1) / (s2 - s1)
    projected = m2 + mb_per_second * (600 - s2)
    assert projected < 4096, (
        f"projected peak RSS for a 10-min upload is {projected:.0f} MB "
        f"({mb_per_second:.1f} MB per audio-second); target is < 4096 MB"
    )
    assert baseline > 0  # keep the baseline measurement referenced for clarity
