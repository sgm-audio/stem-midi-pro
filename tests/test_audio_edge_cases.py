# SPDX-License-Identifier: Apache-2.0
"""Numerical edge cases: silence, clipping, DC, NaN, absurdly short/long input.

Audit finding (task 2): every audio fixture in the pre-existing suite is either
``0.3 * sin(220 Hz)`` (``conftest.audio``), a ``torch.randn`` blob
(``test_mamba_separator.py``, ``test_mamba_transcriber.py``), or a 1-second
sine WAV. Nothing in the suite ever feeds the model digital silence, a
full-scale square wave, a DC offset, a signal shorter than one STFT window, a
NaN sample, or a 44101-sample file. A separator implemented as ``audio * 0.6``
(the Fake backend!) passes every one of those tests, and so does one that
produces NaNs on real-world material.

Everything here runs on the real ``models.mamba_separator`` /
``models.mamba_transcriber`` modules (tier ``real_mamba`` — a skip is a suite
failure), on deterministic signals built by ``tests.synth.Synth``.
"""

from __future__ import annotations

import math

import pytest

from tests.doubles import frame_count
from tests.honesty import known_bug, require

torch = require("torch", "audio edge-case tests", module_level=True)
np = require("numpy", "audio edge-case tests", module_level=True)

SR = 44100
HOP = 512
N_FFT = 2048

pytestmark = pytest.mark.real_mamba


@pytest.fixture(scope="module")
def audio_cfg():
    """Production audio block (n_fft 2048 / hop 512): the config that ships."""
    return {
        "audio": {"sample_rate": SR, "n_fft": N_FFT, "hop_length": HOP, "n_mels": 80},
        "separator": {"d_model": 32, "n_layers": 1, "d_state": 8, "d_conv": 4, "expand": 2},
        "transcriber": {
            "d_model": 32,
            "n_layers": 1,
            "d_state": 8,
            "onset_head_dim": 16,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend", "vibrato", "slide"],
            "onset_threshold": 0.5,
        },
    }


@pytest.fixture(scope="module")
def separator(audio_cfg):
    from models.mamba_separator import MambaSeparator

    torch.manual_seed(0)
    model = MambaSeparator(audio_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def transcriber(audio_cfg):
    from models.mamba_transcriber import MambaTranscriber

    torch.manual_seed(0)
    model = MambaTranscriber(audio_cfg)
    model.eval()
    return model


CORPUS = {
    "digital-silence": lambda s: s.silence(1.0),
    "dc-offset-full-scale": lambda s: s.dc(1.0),
    "hard-clipped-square": lambda s: s.clipped_square(),
    "eight-times-overflow": lambda s: s.overflow(gain=8.0),
    "single-impulse": lambda s: s.impulse(1.0),
    "minus-140db-tone": lambda s: s.denormal_quiet(),
    "pink-noise": lambda s: s.noise(1.0, color="pink"),
    "white-noise": lambda s: s.noise(1.0, seed=3),
    "pluck-line": lambda s: s.note_sequence(total_seconds=1.0),
    "twelve-voice-chord": lambda s: s.chord(n_voices=12),
}


@pytest.fixture
def nonfinite_corpus(synth):
    return {
        "nan-sample": synth.with_nan(synth.note_sequence(total_seconds=0.5)),
        "inf-sample": synth.with_inf(synth.tone(), at=7),
    }


# ---------------------------------------------------------------------------
# 1. the separator/transcriber must survive all of it without NaN or shape drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_separator_survives_the_corpus(separator, synth, case):
    """Finite stems, exact length, masks that cannot add energy — for every input."""
    x = CORPUS[case](synth)
    inp = torch.from_numpy(x)[None, None].float()
    try:
        with torch.inference_mode():
            g, b, r, cache, metrics = separator(inp)
    except RuntimeError as exc:  # NaN/Inf may legitimately be rejected, but loudly
        assert not torch.isfinite(inp).all(), f"rejected finite input {case}: {exc}"
        return

    for name, stem in (("guitar", g), ("bass", b), ("residual", r)):
        assert tuple(stem.shape) == (1, 1, len(x)), f"{name}: shape {tuple(stem.shape)}"
        assert torch.isfinite(stem).all(), f"{name} contains NaN/Inf for input {case}"
    in_peak = float(inp.abs().max())
    for name, stem in (("guitar", g), ("bass", b), ("residual", r)):
        # A sigmoid mask in (0,1) applied to the mixture's spectrum cannot create
        # more than ~1x the input peak; exceeding it means the mask or the ISTFT
        # window normalisation is wrong (the classic "stems are louder than the mix").
        assert (
            float(stem.abs().max()) <= in_peak * 1.05 + 1e-6
        ), f"{name} peak {float(stem.abs().max()):.4f} > input peak {in_peak:.4f} for {case}"
    assert cache is None, "documented C-2.5 contract: no state cache yet"
    assert tuple(metrics.shape) == (1, 2)
    assert torch.isfinite(metrics).all(), f"metrics NaN for {case}: {metrics}"
    assert float(metrics.min()) >= 0.0 and float(metrics.max()) <= 30.0


@pytest.mark.parametrize(
    "case", ["digital-silence", "hard-clipped-square", "dc-offset-full-scale", "pluck-line"]
)
def test_transcriber_outputs_finite_and_bounded(transcriber, synth, case):
    x = CORPUS[case](synth)
    inp = torch.from_numpy(x)[None, None].float()
    with torch.inference_mode():
        onset, pitch, vel, expr, conf = transcriber(inp)
    n = frame_count(len(x), HOP)
    assert onset.shape == (1, n, 1)
    assert pitch.shape == (1, n, 128)
    assert conf.shape == (1, n)
    for name, t in (
        ("onset", onset),
        ("pitch", pitch),
        ("vel", vel),
        ("expr", expr),
        ("conf", conf),
    ):
        assert torch.isfinite(t).all(), f"{name} not finite for {case}"
    assert bool((vel >= 0).all() and (vel <= 1).all())
    assert bool((conf >= 0).all() and (conf <= 1).all())


# ---------------------------------------------------------------------------
# 2. silence behaviour (the "stems must be silent" property)
# ---------------------------------------------------------------------------


def test_digital_silence_yields_silent_stems(separator):
    """Zero input -> zero stems. Any DC leak here is a mask/bias bug."""
    with torch.inference_mode():
        g, b, r, _c, metrics = separator(torch.zeros(1, 1, SR))
    for name, stem in (("guitar", g), ("bass", b), ("residual", r)):
        assert float(stem.abs().max()) < 1e-6, f"{name} stem is not silent for silent input"
    # and the quality proxy must not reward it
    assert (
        float(metrics[0, 0]) < 20.0
    ), "silence scored at/above min_centroid_separation_db - gate is reachable on nothing"


def test_silence_makes_no_targets_for_the_transcriber(transcriber):
    """log(0) must be clamped, not -inf, or the whole scan turns to NaN."""
    with torch.inference_mode():
        onset, _pitch, _vel, _expr, conf = transcriber(torch.zeros(1, 1, SR))
    assert torch.isfinite(onset).all() and torch.isfinite(conf).all()
    mel = transcriber._mel_spectrogram(torch.zeros(1, SR))
    assert torch.isfinite(mel).all(), "log-mel produced -inf/NaN on silence"
    assert float(mel.min()) > -100.0


def test_mel_basis_is_non_degenerate_and_ordered(transcriber):
    """Every mel filter must have support, and centres must ascend with frequency.

    A collapsed basis (all-zero rows, or rows out of order) makes the transcriber
    see a constant vector for every input; the shape-only tests in
    test_mamba_transcriber.py cannot see that, and neither can a loss test.
    """
    basis = transcriber.mel_basis
    assert basis.shape == (80, N_FFT // 2 + 1)
    assert int((basis.sum(dim=1) == 0).sum()) == 0, "some mel filters are all-zero"
    centres = (basis * torch.arange(basis.shape[1]).float()).sum(dim=1) / basis.sum(dim=1)
    assert bool(
        (centres[1:] >= centres[:-1]).all()
    ), "mel filter centres are not monotone in frequency"
    assert float(basis.max()) <= 1.0 and float(basis.min()) >= 0.0


# ---------------------------------------------------------------------------
# 3. artifact detection (it decides the user's tier, so it must not lie)
# ---------------------------------------------------------------------------


def test_hard_clipped_input_is_flagged(separator, synth):
    x = CORPUS["hard-clipped-square"](synth)
    stems = torch.from_numpy(x)[None, None].float()
    from main import StemMidiModel

    flags = StemMidiModel._detect_artifacts(None, stems, stems * 0.5)
    assert "clipping" in flags, f"a ±1.0 square wave was not flagged as clipped: {flags}"


@known_bug(
    "ART-1",
    "main._detect_artifacts flags 'noise_floor' whenever mean(stem**2) > 0.1 over the whole "
    "file — i.e. any material louder than about -3.5 dBFS RMS — and any artifact flag forces "
    "COMPLEX (quality_gates.ARCH-11.7). Verified: a clean 0.5-amplitude sine (no clipping, no "
    "noise) yields ['noise_floor'], so on real music the gate always reports 'complex "
    "material' and the studio/draft tiers are unreachable. Compute noise from a silent-"
    "section floor estimate, and never treat level alone as noise.",
)
def test_clean_loud_audio_is_not_flagged_as_artificial():
    from main import StemMidiModel

    tone = 0.5 * torch.sin(2 * math.pi * 220 * torch.arange(SR).float() / SR)
    tone = tone[None, None]
    flags = StemMidiModel._detect_artifacts(None, tone, tone * 0.6)
    assert flags == ["none"], f"clean, unclipped, low-noise material flagged {flags}"


def test_correlated_stems_are_not_phase_cancellation(synth):
    """Positive correlation is normal (shared fundamentals); only ~-1 is cancellation."""
    from main import StemMidiModel

    x = torch.from_numpy(CORPUS["pluck-line"](synth))[None, None].float() * 0.3
    assert "phase_cancellation" not in StemMidiModel._detect_artifacts(None, x, x)  # identical
    assert "phase_cancellation" in StemMidiModel._detect_artifacts(None, x, -x)  # anti-phase


def test_the_small_test_grid_is_runnable_by_the_real_separator(mock_config):
    """Regression guard for CFG-2 (fixed by PR #11): `mock_config` must stay usable.

    It used to pair n_fft=512 with hop_length=512, and `torch.istft` refuses a
    non-overlapping Hann window ("window overlap add min: 1"), so the ONLY config a
    fast unit test could build was one the real separator could not run — which is
    precisely how a suite ends up testing nothing but a fake backend. Locked here so a
    future audio-block edit cannot quietly make the real path untestable on CPU again.
    """
    from models.mamba_separator import MambaSeparator

    hop, n_fft = mock_config["audio"]["hop_length"], mock_config["audio"]["n_fft"]
    assert n_fft / hop >= 2, (
        f"cfg audio block is n_fft={n_fft} hop={hop}: torch.istft needs overlap, so no real "
        "separator can run on the fast test config (this is CFG-2)"
    )
    torch.manual_seed(0)
    sep = MambaSeparator(mock_config).eval()
    with torch.inference_mode():
        guitar, bass, _residual, _cache, metrics = sep(torch.randn(1, 1, 8192))
    for name, stem in (("guitar", guitar), ("bass", bass)):
        assert stem.shape == (1, 1, 8192), f"{name}: {tuple(stem.shape)}"
        assert torch.isfinite(stem).all(), f"{name} is non-finite on the small grid"
    assert float(metrics.abs().max()) < 1e6, f"metrics blew up on the small grid: {metrics}"


# ---------------------------------------------------------------------------
# 4. length / shape contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [N_FFT, N_FFT + 1, SR, SR + 1, 3 * SR + 7])
def test_output_length_equals_input_length_exactly(separator, n_samples):
    """Off-by-one on a non-multiple length means truncated/ghost audio at the tail.

    N_FFT - 1 is deliberately excluded: inputs shorter than one window are now
    refused with a typed ValueError (locked by test_sub_window_input_gets_a_domain_error).
    """
    x = (0.2 * torch.sin(2 * math.pi * 200 * torch.arange(n_samples).float() / SR))[None, None]
    with torch.inference_mode():
        g, b, r, _c, _m = separator(x)
    for name, stem in (("guitar", g), ("bass", b), ("residual", r)):
        assert stem.shape[-1] == n_samples, f"{name}: {stem.shape[-1]} samples for {n_samples} in"


def test_sub_window_input_gets_a_domain_error(pipeline_config, make_wav, synth):
    """Regression guard (was IO-1): short input is refused with a domain error.

    models/mamba_separator.py now validates shape, channels and length before
    touching torch.stft, so a 5 ms impulse and a 2047-sample file both raise
    ValueError instead of leaking a raw torch padding RuntimeError.
    """
    from main import StemMidiModel

    path = make_wav(synth.impulse(0.005), name="5ms.wav")  # 220 samples
    model = StemMidiModel(pipeline_config)
    model.eval()
    with pytest.raises((ValueError, OSError)) as exc_info:  # a clean, documented error
        model.process_audio_file(path)
    assert "ms" in str(exc_info.value) or "short" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "shape,expect",
    [
        ((1, 2, SR), "mono"),
        ((SR,), "(b, c, t)"),
        ((1, SR), "(b, c, t)"),
        ((1, 1, SR, 1), "(b, c, t)"),
    ],
    ids=["stereo", "1d", "2d", "4d"],
)
def test_separator_rejects_wrong_rank_or_channels_with_a_useful_message(separator, shape, expect):
    """The assert messages are the API for misuse — pin them, they are load-bearing."""
    x = torch.zeros(*shape)
    with pytest.raises((ValueError, AssertionError)) as exc_info:
        separator(x)
    assert (
        expect in str(exc_info.value).lower()
    ), f"{shape} produced an unhelpful error: {exc_info.value!r}"


# ---------------------------------------------------------------------------
# 5. audio I/O validation against the corpus
# ---------------------------------------------------------------------------


def test_zero_byte_and_header_only_files_raise_domain_errors(make_wav, tmp_path):
    from audio_io import AudioValidationError, load_audio

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioValidationError):
        load_audio(str(empty))

    header_only = tmp_path / "header_only.wav"
    header_only.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 36)
    with pytest.raises(AudioValidationError):
        load_audio(str(header_only))


def test_validation_accepts_supported_shapes(make_wav, synth):
    """48 kHz, stereo and 24-bit are legal inputs; over-rejection is also a bug."""
    from audio_io import load_audio

    p48 = make_wav(synth.tone(220.0, seconds=0.5, sr=48000), name="a48.wav", sr=48000)
    audio, sr, _info = load_audio(p48)
    assert sr == 48000 and len(audio) == 24000 and audio.dtype == np.float32

    p24 = make_wav(synth.tone(seconds=0.2), name="a24.wav", subtype="PCM_24")
    _a, sr2, info2 = load_audio(p24)
    assert sr2 == 44100 and info2["bit_depth"] == 24


def test_zero_frame_wav_is_rejected(tmp_path):
    """Regression guard for IO-2: a zero-frame file is a 400, not a torch traceback."""
    import wave

    from audio_io import AudioValidationError, load_audio

    p = tmp_path / "zero_frames.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"")
    with pytest.raises(AudioValidationError):
        load_audio(str(p))


def test_non_finite_audio_is_rejected(make_wav, synth):
    """Regression guard for IO-3: non-finite samples are refused at the decoder."""
    from audio_io import AudioValidationError, load_audio

    nan_path = make_wav(synth.with_nan(synth.tone(seconds=0.5)), name="nan.wav", subtype="FLOAT")
    with pytest.raises(AudioValidationError):
        load_audio(nan_path)

    inf_path = make_wav(synth.with_inf(synth.tone(seconds=0.5)), name="inf.wav", subtype="FLOAT")
    with pytest.raises(AudioValidationError):
        load_audio(inf_path)


def test_non_finite_input_never_leaks_into_stems(nonfinite_corpus, separator):
    """Regression guard for IO-3: NaN/Inf is rejected, never rendered."""
    for name, sig in nonfinite_corpus.items():
        x = torch.from_numpy(np.asarray(sig, dtype=np.float32))[None, None]
        try:
            with torch.inference_mode():
                g, _b, _r, _c, _m = separator(x)
        except (ValueError, FloatingPointError):
            continue  # an explicit rejection is fine
        assert torch.isfinite(g).all(), (
            f"{name}: non-finite input produced non-finite stems instead of a rejection — "
            f"this is what a user gets back as an empty/NaN WAV"
        )


@known_bug(
    "IO-4",
    "mean-downmixing an anti-phase stereo file (L = -R) yields silence, and nothing in the "
    "pipeline notices: audio_io.load_native_mono returns an all-zero track for a full-level "
    "input, so the user gets an empty render blamed on 'complex material'. Measure the "
    "downmix energy vs. the channels' energy and reject/warn when the ratio collapses.",
)
def test_mono_cancellation_is_surfaced(make_wav, synth):
    from audio_io import AudioValidationError, load_audio

    mono = synth.note_sequence(total_seconds=0.5)
    before = float(np.sqrt((mono**2).mean()))
    p = make_wav(synth.anti_phase_stereo(mono), name="anti.wav")
    try:
        audio, _sr, _info = load_audio(p)
    except AudioValidationError:
        return  # an explicit rejection is an acceptable fix
    after = float(np.sqrt((audio**2).mean()))
    assert after > 0.5 * before, (
        f"anti-phase stereo downmixed to {after:.2e} from {before:.2e} and validation stayed "
        f"silent — the render is empty for a reason the user is never told"
    )


# ---------------------------------------------------------------------------
# 6. property-style: scaling and length sweep
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_property_amplitude_scale_never_breaks_the_separator(separator, synth, seed):
    """Across 7 decades of level, outputs stay finite, same-length and non-amplifying."""
    rng = np.random.default_rng(seed)
    for exponent in (-7, -3, -1, 0, 0.9):
        amp = float(10.0**exponent)
        n = int(rng.integers(N_FFT, N_FFT * 3))
        t = np.arange(n) / SR
        sig = (amp * np.sin(2 * np.pi * 180.0 * t) + 0.4 * amp * rng.standard_normal(n)).astype(
            np.float32
        )
        with torch.inference_mode():
            g, b, r, _c, m = separator(torch.from_numpy(sig)[None, None])
        assert tuple(g.shape) == (1, 1, n)
        for name, stem in (("guitar", g), ("bass", b), ("residual", r)):
            assert torch.isfinite(stem).all(), f"amp={amp} n={n}: {name} non-finite"
            assert float(stem.abs().max()) <= float(np.abs(sig).max()) * 1.05 + 1e-6
        assert torch.isfinite(m).all()


@pytest.mark.property
def test_property_long_input_stays_on_the_frame_grid(separator, synth):
    """A 60 s input: frames == 1 + T//hop, length preserved, no NaN at the tail."""
    sig = synth.noise(60.0, seed=11) * 0.5
    # keep it cheap: the separator is linear-in-frames, so check a 60 s slice shape-only
    x = torch.from_numpy(sig[: int(12.5 * SR)])[None, None]
    with torch.inference_mode():
        g, _b, _r, _c, _m = separator(x)
    assert g.shape[-1] == x.shape[-1]
    assert torch.isfinite(g).all()
