# Content/register separation: new multi-split results

## Status and provenance

A synthetic structured-input experiment, not pretrained fine-tuning, free-form language evaluation, or evidence of improved human understanding. The locally recorded pre-run protocol is `PREREGISTRATION.md`, SHA256 `3838c84924f98a807087fa1d87d6a690e9455372d7047b7fff6d9e86b07014e8`. It was written before full runs but is not externally registered or independently timestamped. Existing pilot results motivated this design.

Protocol history (local conversation/tool timestamps, UTC): initial version `PROTOCOL.initial.md` was recorded at 04:17:44 with SHA256 `01208efe530ca29e1d0c2cf56a17c3f538bb31bd816fcad127337bc79d37c2b8`. Before the full experiment, by 04:18:26, the heading was changed from “preregistration” to “locally recorded pre-run protocol” and the lack of independent timestamp/external registration was explicitly added. No hypotheses or design changed. The resulting `3838…` hash was printed at full-run launch at 04:18:54; first results arrived at 04:19:07. Both versions are retained. These are local execution records, not independent registration evidence.

Post-run accounting correction: the protocol counted 1164 nonauxiliary joint parameters for the unconditioned model, but 96 weights receive permanently zero register inputs. Its effective active count is 1068; structural nonaux count remains 1164. All joint models allocate 1263 parameters including the unused 99-parameter auxiliary head in nonauxiliary objectives. Results JSON distinguishes structural and effective active counts. This correction changes no model, training, split, score, or selection. The original protocol has not been rewritten.

## Results

Equal 100 full-batch updates, three dataset splits and three initialization seeds each. Mean of nine runs, without an independence/significance claim:

| Objective | Effective active parameters | Answer accuracy | Answer-change rate |
|---|---:|---:|---:|
| Unconditioned joint |1068|99.57%|0.00%|
| Conditioned joint |1164|77.86%|19.10%|
| Auxiliary joint |1263|74.88%|20.75%|
| Auxiliary + consistency |1263|74.68%|19.36%|
| Separated width 32 |804|98.96%|0.00%|
| Separated width 46 |1154|99.31%|0.00%|

Separated width 32 versus conditioned: +21.09 percentage points mean, range +11.20 to +26.82 across paired runs;9 wins,0 ties,0 losses. Width 46: +21.44 points, range +13.54 to +26.82;9 wins,0 ties,0 losses. Full per-run data and paired differences are in `results.json`.

Crucially, the unconditioned control is slightly more accurate than either separated model. This does not show an accuracy benefit over that simpler content-invariant model. A deterministic requested-register formatter could also be applied to its predicted answer. The finding is that allowing presentation requests to influence content was harmful in this toy design, not that a novel learning method universally improves language.

## What is and is not preserved

The content branch sees only the 20 features encoding only the source record and query. The formatter sees the model-predicted answer and explicit requested register, never the gold answer or source record. It changes only fixed wording. Answer invariance across register requests follows structurally; wrong answers remain wrong. Tests check every answer/register template preserves exactly the predicted integer, including deliberately wrong predictions. Deterministic template-choice agreement is 1.0 by construction, not learned register accuracy or a comprehension metric.

Widths 32 and 46 separate width from approximate parameter matching. Width 46 has 1154 active parameters versus conditioned 1164; equal update counts and near-equal parameters are not equal FLOPs, wall time, initialization or optimization difficulty. Separated validation selects by answer cross-entropy, joint models by joint cross-entropy. Source records are disjoint within each split, but the three splits overlap the same 256-record universe; the four queries per record are correlated. No statistical significance or general-domain extrapolation is claimed.

## Reproduce

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m run_separation --out /tmp/separation-results.json
cmp milestone2/results.json /tmp/separation-results.json
sha256sum -c milestone2/MANIFEST.sha256
```

Installed NumPy 2.3.5 and Python 3.12.14 were used. No downloads or API calls. Two complete executions matched exactly. The earlier baseline remains unchanged. Next work should test richer outputs, independently authored content, explicit semantic constraints and real human outcomes before claiming natural-language benefit.
