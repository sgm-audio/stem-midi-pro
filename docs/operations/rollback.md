# Rollback — Stem+MIDI Pro

How to revert a bad deploy on a small CPU deployment.

## Code rollback

```bash
cd /path/to/stem-midi-pro
git checkout <previous-tag>            # or: git revert <bad-commit> on main
source venv/bin/activate
pip install -r requirements.txt        # restore older deps if they changed
sudo systemctl restart stem-midi-pro
curl localhost:8000/health             # expect 200
```

Verify with one known test file: compare `processing_report.json` against the pre-upgrade baseline.

## Model-weights rollback

1. Point `MODEL_CHECKPOINT_PATH` back to the previous checkpoint (you kept it, per [upgrading.md](upgrading.md)).
2. Restart; re-run the test file; confirm metrics returned to baseline.
3. Keep a "last known good" checkpoint permanently; never overwrite it.

## When to roll back (triggers)

- `/health` returns non-200 after restart.
- `500` error rate spikes on `/process` for files that previously succeeded.
- Memory usage jumps beyond the machine's headroom (OOM kills).
- Quality tiers collapse (e.g. everything routes to `complex` vs. historical mix).

## Notes

- No user data is at risk: uploads are ephemeral and there is no database; rollback never needs data restore.
- If the rollback itself fails to start, check `configs/model_config.yaml` compatibility — revert config to the version that matches the rolled-back code.
- Record the incident (symptom, versions, resolution) in the changelog or issue tracker.
