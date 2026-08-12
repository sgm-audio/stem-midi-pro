#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Smoke eval: run StemMidiModel on a wav and print report + event counts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer repo root over any site-packages module named ``main``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--config", default="configs/model_config.cpu.yaml", type=Path)
    args = p.parse_args()

    if not args.audio.is_file():
        print(f"Audio not found: {args.audio}", file=sys.stderr)
        return 2

    cfg_path = args.config if args.config.is_absolute() else _ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from main import StemMidiModel

    model = StemMidiModel(cfg)
    model.eval()
    result = model.process_audio_file(str(args.audio))

    report = result["report"]
    g_events = result.get("midi_guitar", {}).get("midi_events", [])
    b_events = result.get("midi_bass", {}).get("midi_events", [])
    print(f"quality_tier={report.quality_tier.value}")
    print(f"si_sdr_proxy={report.si_sdr:.2f}")
    print(f"avg_confidence={report.avg_confidence:.3f}")
    print(f"artifacts={report.artifact_flags}")
    print(f"guitar_notes={len(g_events)} bass_notes={len(b_events)}")
    print(f"routing={result['routing']['action']}")
    g = result["stems"]["guitar"]
    print(f"guitar_stem_shape={g.shape} peak={float(abs(g).max()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
