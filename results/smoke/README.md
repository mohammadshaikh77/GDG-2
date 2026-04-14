# Smoke Results (Committed Proof Artifacts)

These files are lightweight smoke-run outputs committed intentionally so repository reviewers can verify that the train → measure → plot pipeline executes end-to-end.

## Files

- `exp1_smoke_seed0_log.csv` — training log from smoke run
- `exp1_smoke_seed0_step500.pt` — checkpoint at step 500
- `exp1_smoke_seed0_step500_metrics.npz` — measured metrics from checkpoint
- `exp1_smoke_vocab.json` — vocabulary mapping used in smoke run

## Notes

- These are demonstration artifacts, not final experiment-scale results.
- Full datasets and large-scale outputs remain excluded from git.
