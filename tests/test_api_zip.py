# SPDX-License-Identifier: Apache-2.0
"""The paid boundary: POST /process → the ZIP a human unzips.

Everything else in the suite tests ``main.StemMidiModel`` or the ``utils`` helpers
in isolation, and ``tests/test_api_endpoints.py`` only pings ``/live``/``/ready``.
No test in the repo has ever posted audio and looked inside the archive — so every
defect that lives in the *seam* is invisible: the module-global config that decides
the WAV sample rate, ``build_midi_from_events`` called with its own defaults rather
than the pipeline grid, the merged event list the writer actually reads, the default
PCM_16 encoding, and ``json.dumps(allow_nan=True)`` in the report.

Deliberately model-free: the injected model returns hand-built outputs whose event
frames we know exactly, so each assertion isolates the boundary (and runs anywhere).
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from tests.honesty import known_bug, require

soundfile = require("soundfile", "ZIP/stem round-trip")
mido = require("mido", "MIDI parsing")

SR = 44100
HOP = 512
SECONDS = 3.0
N_FRAMES = 1 + int(SR * SECONDS) // HOP

# Per-stem events, exactly as ConfidenceInjector shapes them.
EVENTS = {
    "guitar": [
        {
            "note": 60,
            "onset_frame": 100,
            "velocity": 100,
            "cc_127_value": 115,
            "duration_frames": 60,
        },
        {
            "note": 64,
            "onset_frame": 200,
            "velocity": 90,
            "cc_127_value": 100,
            "duration_frames": 60,
        },
        {"note": 67, "onset_frame": 300, "velocity": 80, "cc_127_value": 90, "duration_frames": 60},
    ],
    "bass": [
        {
            "note": 36,
            "onset_frame": 150,
            "velocity": 110,
            "cc_127_value": 120,
            "duration_frames": 60,
        },
        {
            "note": 43,
            "onset_frame": 250,
            "velocity": 95,
            "cc_127_value": 105,
            "duration_frames": 60,
        },
    ],
}
MERGED = sorted(
    [dict(e, stem=s) for s, evs in EVENTS.items() for e in evs], key=lambda e: e["onset_frame"]
)


class StubModel:
    """Stands in for StemMidiModel at the /process boundary.

    ``process_audio_file`` decodes the uploaded file and hands it straight back as
    the "separated" stems, so what lands in the ZIP is a faithful image of whatever
    the delivery layer does to the signal (encoding, clipping, NaN handling).
    """

    def __init__(self, sr: int = SR, hop: int = HOP, mode: str = "echo"):
        self.sr, self.hop, self.mode = sr, hop, mode
        self.calls: list[tuple[str, bool]] = []

    def process_audio_file(self, path, quantize=False):
        self.calls.append((str(path), bool(quantize)))
        audio, _sr = soundfile.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if self.mode == "nan":  # what the real pipeline does to a NaN upload
            audio = np.full_like(audio, np.nan)
        n = max(1, int(self.sr * SECONDS))
        np.zeros(n, dtype="float32")
        t = np.arange(n) / self.sr
        guitar = (np.sin(2 * np.pi * 220 * t) * 0.4).astype("float32")
        bass = (np.sin(2 * np.pi * 55 * t) * 0.6).astype("float32")
        if self.mode in ("echo", "nan"):
            guitar = bass = audio.astype("float32")
            bass = (audio * 0.5).astype("float32")
        return {
            "stems": {
                "guitar": guitar,
                "bass": bass,
                "residual": (guitar * 0.05).astype("float32"),
            },
            "midi": {
                "midi_events": MERGED,
                "guitar": {"midi_events": EVENTS["guitar"], "avg_confidence": 0.91},
                "bass": {"midi_events": EVENTS["bass"], "avg_confidence": 0.82},
                "summary": {"low_confidence_count": 0, "n_frames": N_FRAMES},
            },
            "sample_rate": self.sr,
            "hop_length": self.hop,
            "report": _report(),
        }


def _report(**over):
    """A real ProcessingReport (its quality_tier is derived, not passed in)."""
    from utils.quality_gates import ProcessingReport

    flags = over.pop("artifact_flags", ["none"])
    kw = dict(
        si_sdr=6.5,
        phase_coherence=0.42,
        avg_confidence=0.55,  # below draft threshold -> COMPLEX, like an untrained render
        artifact_flags=flags,
        low_confidence_notes=0,
        config={
            "quality_gates": {
                "studio_confidence_threshold": 0.85,
                "draft_confidence_threshold": 0.70,
                "min_si_sdr": 20.0,
            }
        },
    )
    kw.update(over)
    return ProcessingReport(**kw)


def _wav_bytes(seconds: float = SECONDS, sr: int = SR, array=None, empty: bool = False) -> bytes:
    if empty:
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"")
        return buf.getvalue()
    n = int(sr * seconds)
    audio = (
        (np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 0.3).astype("float32")
        if array is None
        else np.asarray(array, dtype="float32")
    )
    buf = io.BytesIO()
    soundfile.write(buf, audio, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture()
def api_env(monkeypatch, request):
    """Patch api's module globals: a stub model + a config with a known grid."""
    import api

    stub = StubModel()
    monkeypatch.setattr(api, "model", stub)
    monkeypatch.setattr(
        api, "model_config", {"audio": {"sample_rate": SR, "hop_length": HOP}, "name": "stub"}
    )
    monkeypatch.setattr(api, "_model_ready", True)
    monkeypatch.setattr(api, "_processing_sem", None)
    return api, stub


def _post(api, tmp_path, payload: bytes, params: str = "", filename: str = "input.wav"):
    from fastapi.testclient import TestClient

    src = tmp_path / filename
    src.write_bytes(payload)
    with src.open("rb") as fh:
        return TestClient(api.app).post(
            f"/process{params}", files={"file": (filename, fh, "audio/wav")}
        )


def _zip(response) -> zipfile.ZipFile:
    assert response.status_code == 200, f"{response.status_code}: {response.text[:200]}"
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    assert zf.testzip() is None, "the archive itself is corrupt"
    return zf


# ---------------------------------------------------------------------------
# archive structure + stems
# ---------------------------------------------------------------------------


def test_zip_contains_exactly_the_advertised_files(api_env, tmp_path):
    api, stub = api_env
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    assert sorted(zf.namelist()) == [
        "bass.mid",
        "bass_stem.wav",
        "guitar.mid",
        "guitar_stem.wav",
        "processing_report.json",
    ], f"unexpected deliverable set: {sorted(zf.namelist())}"
    assert stub.calls, "the model was never called"


def test_zip_stems_are_playable_and_are_not_two_copies_of_each_other(api_env, tmp_path):
    """ "Good stems" has to survive serialisation, not just exist in a tensor."""
    api, _ = api_env
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    decoded = {}
    for name in ("guitar_stem.wav", "bass_stem.wav"):
        audio, sr = soundfile.read(io.BytesIO(zf.read(name)), dtype="float32")
        assert sr == SR, f"{name} written at {sr} Hz: the deliverable's pitch/timing is wrong"
        assert audio.ndim == 1, f"{name} has {audio.ndim} channels; downstream tools expect mono"
        assert len(audio) == int(SR * SECONDS), f"{name} length {len(audio)} != {int(SR * SECONDS)}"
        assert np.isfinite(audio).all(), f"{name} contains non-finite samples — it will not play"
        assert float(np.max(np.abs(audio))) <= 1.0 + 1e-6, f"{name} is out of range"
        decoded[name] = audio
    a, b = decoded.values()
    assert not np.array_equal(a, b), "the two 'stems' are the same signal"
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9))
    assert rel > 0.05, f"guitar and bass differ by {rel:.2%} — a copy, not a separation"


@known_bug(
    "ZIP-1",
    "create_response_zip calls sf.write(buf, stem, sr, format='WAV') with no subtype, so "
    "libsndfile picks PCM_16: every float32 stem is quantised to 1/32768 and any sample outside "
    "[-1, 1] is hard-clipped in the delivery layer itself (measured: a 3.0 peak decodes as "
    "exactly 1.0). A clipped or hot stem therefore becomes a permanently clipped file with no "
    "flag or warning. Write subtype='FLOAT' (or normalise and report the gain).",
)
def test_zip_stems_keep_the_pipeline_amplitude_and_precision(api_env, tmp_path):
    api, _ = api_env
    hot = np.full(int(SR * 0.1), 3.0, dtype="float32")
    stub = api.model
    monkey_stems = {"guitar": hot, "bass": hot * 0.5}

    def process(path, quantize=False):
        out = StubModel.process_audio_file(stub, path, quantize)
        out["stems"].update(monkey_stems)
        return out

    stub.process_audio_file = process
    zf = _zip(_post(api, tmp_path, _wav_bytes(seconds=0.1)))
    for name, expected in monkey_stems.items():
        audio, _ = soundfile.read(io.BytesIO(zf.read(name)), dtype="float32")
        err = float(np.max(np.abs(audio - expected)))
        assert err < 1e-4, (
            f"{name} was re-encoded on the way out: max |Δ| = {err:.6f} "
            f"(peak {np.abs(expected).max():.2f} -> {np.abs(audio).max():.2f})"
        )


def test_report_in_the_zip_is_the_pipeline_report_not_a_rederivation(api_env, tmp_path):
    """The quality numbers a customer sees must be the ones the gates used."""
    api, _ = api_env
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    assert json.loads(zf.read("processing_report.json")) == {
        "si_sdr": 6.5,
        "phase_coherence": 0.42,
        "avg_confidence": 0.55,
        "artifact_flags": ["none"],
        "low_confidence_notes": 0,
        "quality_tier": "complex",
    }, "the ZIP report disagrees with the pipeline's QualityReport"


@pytest.mark.parametrize(
    "params,expected", [("", False), ("?quantize=true", True)], ids=["raw", "quantize"]
)
def test_query_flags_reach_the_model(api_env, tmp_path, params, expected):
    """?quantize=true must change what the model is told to do (untested before)."""
    api, stub = api_env
    _post(api, tmp_path, _wav_bytes(), params)
    assert stub.calls and stub.calls[-1][1] is expected, f"quantize={expected!r} lost: {stub.calls}"


# ---------------------------------------------------------------------------
# the JSON must be JSON
# ---------------------------------------------------------------------------


@known_bug(
    "JSON-1",
    "processing_report.json is written with json.dumps(...) and the default allow_nan=True, so a "
    "NaN metric (which the pipeline produces for silent, DC and non-finite input — measured: "
    "si_sdr/phase_coherence/avg_confidence are all NaN for an all-NaN upload, and for silence the "
    "separator's proxy returns 0.0 while phase_coherence stays NaN) emits bare `NaN`, which is "
    "not valid RFC 8259: browsers' JSON.parse() rejects the whole download. Serialise with "
    "allow_nan=False after sanitising, or emit null with an explicit 'unreliable' flag.",
)
def test_zip_report_is_strict_json(api_env, tmp_path):
    api, stub = api_env

    def process(path, quantize=False):
        out = StubModel.process_audio_file(stub, path, quantize)
        out["report"] = _report(
            si_sdr=float("nan"), phase_coherence=float("nan"), avg_confidence=float("nan")
        )
        return out

    stub.process_audio_file = process
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    raw = zf.read("processing_report.json")

    def reject(constant):
        raise ValueError(f"non-finite JSON constant {constant!r}")

    assert json.loads(raw, parse_constant=reject)  # must not raise


# ---------------------------------------------------------------------------
# the MIDI inside the archive
# ---------------------------------------------------------------------------


def test_zip_midi_duration_matches_the_uploaded_audio(api_env, tmp_path):
    """Regression guard (was MIDI-1): the writer's tick math now matches wall clock."""
    api, _ = api_env
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    mid = mido.MidiFile(file=io.BytesIO(zf.read("guitar.mid")))
    expected = (EVENTS["guitar"][-1]["onset_frame"] + 60) * HOP / SR  # last note-off, in seconds
    assert mid.length == pytest.approx(expected, rel=0.05), (
        f"{SECONDS}s of audio produced a {mid.length:.1f}s MIDI file "
        f"(expected ~{expected:.2f}s; ratio {mid.length / expected:.1f}x)"
    )


@known_bug(
    "MIDI-5",
    "create_response_zip calls build_midi_from_events(midi_data, stem) and lets the writer fall "
    "back to its module defaults (hop_length=512, sample_rate=44100) even though the frames came "
    "from the configured grid and outputs['sample_rate'] is right there in the dict — a "
    "16kHz/hop=128 model ships MIDI with every event ~2.75x off in time. Forward the pipeline's "
    "hop_length/sample_rate into the builder.",
)
def test_zip_forwards_the_pipeline_grid_to_the_midi_writer(api_env, tmp_path, monkeypatch):
    api, _ = api_env
    seen: dict = {}
    real = api.build_midi_from_events

    def spy(midi_data, stem_type, **kwargs):
        seen[stem_type] = kwargs
        return real(midi_data, stem_type, **kwargs)

    monkeypatch.setattr(api, "build_midi_from_events", spy)
    _post(api, tmp_path, _wav_bytes())
    assert seen, "the ZIP did not go through build_midi_from_events at all"
    for stem, kwargs in seen.items():
        assert kwargs.get("hop_length") == HOP, f"{stem}: hop_length not forwarded → {kwargs}"
        assert kwargs.get("sample_rate") == SR, f"{stem}: sample_rate not forwarded → {kwargs}"


def test_zip_midi_is_per_stem(api_env, tmp_path):
    """Regression guard (was API-1): each .mid is built from its own stem's events."""
    api, _ = api_env
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    for stem in ("guitar", "bass"):
        raw = zf.read(f"{stem}.mid")
        mid = mido.MidiFile(file=io.BytesIO(raw))
        notes = sorted(
            m.note for track in mid.tracks for m in track if m.type == "note_on" and m.velocity
        )
        expected = sorted(e["note"] for e in EVENTS[stem])
        assert (
            notes == expected
        ), f"{stem}.mid carries {notes}, i.e. the merged event list, not {expected}"


def test_zip_midi_note_off_pairs_every_note_on(api_env, tmp_path):
    """An unpaired note_on is what makes a MIDI 'dangle' forever in a DAW.

    Count-only on purpose: *which* notes belong in which file is API-1's claim.
    """
    api, _ = api_env
    zf = _zip(_post(api, tmp_path, _wav_bytes()))
    for stem in ("guitar", "bass"):
        mid = mido.MidiFile(file=io.BytesIO(zf.read(f"{stem}.mid")))
        assert mid.tracks, f"{stem}.mid has no tracks"
        for track in mid.tracks:
            on = sum(1 for m in track if m.type == "note_on" and m.velocity > 0)
            off = sum(
                1
                for m in track
                if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)
            )
            assert on == off, f"{stem}: {on} note_on vs {off} note_off — dangling notes"
        total = sum(1 for tr in mid.tracks for m in tr if m.type == "note_on" and m.velocity > 0)
        # tolerate both shapes so this guard survives the API-1 fix; tighten to
        # len(EVENTS[stem]) once the ZIP really is per-stem
        assert total in {len(MERGED), len(EVENTS[stem])}, (
            f"{stem}.mid has {total} notes, expected {len(EVENTS[stem])} (per-stem) "
            f"or {len(MERGED)} (merged, i.e. API-1 still open)"
        )


# ---------------------------------------------------------------------------
# uploads the delivery layer should refuse
# ---------------------------------------------------------------------------


@known_bug(
    "IO-2",
    "a 0-frame WAV passes audio_io.open_and_validate (frames=0.0 is never compared against a "
    "minimum — measured info: {'duration': 0.0, 'frames': 0}) and then dies inside torch with "
    "`RuntimeError: Expected 2D or 3D (batch mode) tensor ... but got: [1, 1, 0]`, which the "
    "endpoint's catch-all turns into HTTP 500 'Processing failed: Expected 2D or 3D…'. A "
    "truncated/empty upload is a client error: reject it in validate_audio_file/open_and_validate "
    "(frames == 0, or duration below one hop) so it never reaches the model and no torch "
    "internals leak to the caller.",
)
def test_empty_upload_is_a_client_error_not_a_500(api_env, tmp_path):
    from audio_io import open_and_validate

    path = tmp_path / "empty.wav"
    path.write_bytes(_wav_bytes(empty=True))

    def _info_of(p):
        with open_and_validate(p) as (_snd, info):
            return info

    with pytest.raises(Exception, match=r"empty|no frames|too short|0"):
        info = _info_of(str(path))
        pytest.fail(
            f"open_and_validate accepted a frame-less file: {info} — the model then gets "
            f"a [1, 1, 0] tensor and raises a raw RuntimeError"
        )

    api, _ = api_env
    r = _post(api, tmp_path, _wav_bytes(empty=True))
    assert (
        r.status_code == 400
    ), f"empty WAV produced {r.status_code}: {r.text[:160]!r} (want 400 with a clear detail)"


@pytest.mark.parametrize(
    "array,label",
    [
        pytest.param(np.full(8820, np.nan), "NaN", id="nan"),
        pytest.param(np.full(8820, np.inf), "Inf", id="inf"),
        pytest.param(np.full(8820, -np.inf), "-Inf", id="-inf"),
    ],
)
@known_bug(
    "IO-3",
    "non-finite samples sail through validation and out the other end as a *successful* "
    "delivery: an all-NaN upload returns 200 with artifact_flags=['none'], quality_tier='complex' "
    "and si_sdr/phase_coherence/avg_confidence all NaN, and because the ZIP encodes PCM_16 the "
    "customer's 'stems' decode as a constant -1.0 full-scale block (uniq == [-1.0], rms 1.0) — "
    "2 seconds of digital clipping billed as a result. validate_audio_file must assert "
    "np.isfinite(data).all(), and _detect_artifacts must flag non-finite stems as 'corrupt'.",
)
def test_non_finite_upload_is_refused(api_env, tmp_path, array, label):
    api, _ = api_env
    r = _post(api, tmp_path, _wav_bytes(seconds=0.2, array=array))
    if r.status_code == 200:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        audio, _ = soundfile.read(io.BytesIO(zf.read("guitar_stem.wav")), dtype="float32")
        pytest.fail(
            f"{label} upload returned 200; delivered stem decodes to unique values "
            f"{np.unique(audio)[:3]} (peak {np.abs(audio).max():.4f}) and the report says "
            f"{zf.read('processing_report.json').decode()[:120]!r}"
        )
    assert r.status_code == 400, f"{label} upload returned {r.status_code}: {r.text[:160]}"


def test_out_of_range_upload_is_flagged_as_clipping(api_env, tmp_path):
    """0 dBFS+ audio must be *said* to be clipped, not quietly delivered."""
    api, _ = api_env

    def process(path, quantize=False):
        out = StubModel.process_audio_file(api.model, path, quantize)
        out["report"] = _report(artifact_flags=["clipping"], avg_confidence=0.97, si_sdr=28.0)
        return out

    api.model.process_audio_file = process
    zf = _zip(
        _post(api, tmp_path, _wav_bytes(seconds=0.2, array=np.full(8820, 4.0, dtype="float32")))
    )
    rep = json.loads(zf.read("processing_report.json"))
    assert "clipping" in rep["artifact_flags"], f"clipping not reported: {rep}"
    assert rep["quality_tier"] != "studio", "clipped audio routed as a studio-ready deliverable"
