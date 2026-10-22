# Adaptive Register: recovery rebuild

A runnable NumPy experiment testing explicit presentation-register requests, an auxiliary register head, and counterfactual answer consistency. **This is a newly written recovery implementation, not recovered original source or an open-source language-model fine-tune.** It does not demonstrate improved comprehension or improvement of arbitrary models.

## Run

Python 3.11+ and NumPy are required. Recorded environment: Python 3.12.14, NumPy 2.3.5. No API, download, GPU, or external service is used.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m run --out results/reproduction
sha256sum -c MANIFEST.sha256
```

Existing environments can use their installed NumPy. `requirements.txt` records the tested version; installation is separate. The run regenerates checkpoints, training histories, split keys and metrics. Reference scientific scores are in `results/rebuild/summary.json`. Published text-only checkpoint intentionally excludes binary weights; all 12 models are regenerated locally by the command above.

## Method

Enumerate unique four-slot records with values 0–3; randomly choose 160 using seed 20261022. Split 96/32/32 records before expanding four query slots and three explicit registers. No identical record crosses a split. Inputs contain the four records, queried slot and requested register, encoded as 23 features. A 32-unit tanh encoder predicts a 12-way joint answer/register distribution. A separate auxiliary three-way register head shares the encoder. Full-batch Adam trains 100 epochs; minimum validation joint cross-entropy selects checkpoints. Test scores never select weights.

Four objectives, each run with seeds 11,22,33: unconditioned joint objective; conditioned joint objective; conditioned plus auxiliary weight 0.3; auxiliary plus answer-marginal consistency weight 2.0. Outputs use separate answer and register marginal argmaxes and fixed templates. The consistency term is mean squared answer-probability deviation across three register counterfactuals. It does not guarantee correct answers.

## New results

Mean over three initialization seeds on one fixed split:

| Objective | Answer accuracy | Register agreement | Answer changes across registers | All registers correct |
|---|---:|---:|---:|---:|
| Unconditioned | 100.00% | 33.33% | 0.00% | 100.00% |
| Conditioned | 78.73% | 100.00% | 21.88% | 67.45% |
| Auxiliary | 75.09% | 100.00% | 23.96% | 62.76% |
| Auxiliary + consistency | 74.91% | 100.00% | 22.14% | 63.54% |

The extra head did not improve this toy task. Conditioning caused substantial factual degradation despite perfect register classification. Consistency did not solve it. This negative result is not evidence that the approach will work for natural language. Three seeds on 32 test records do not establish generalization or uncertainty across datasets. The four queries from each record are correlated.

## Verification and scope

Nine tests pass: record-disjoint splitting, counterfactual labels, determinism, finite-difference gradients including consistency, learning, checkpoint round trips, unconditioned invariance, renderer validation, and experiment artifacts. A second complete execution produced byte-identical summaries, splits, histories and all 12 checkpoints. These checks apply only to this new implementation; they do not reproduce the lost original implementation.

Presentation choices are `introductory`, `working`, `technical`; absent choice defaults to `working`. No demographic traits or inferred intelligence are used. No human comprehension, fairness or accessibility study was performed. A real pretrained fine-tune remains unfinished; torch, transformers, peft and pretrained weights were not present or installed for this checkpoint.

## Recovery provenance

Original recovered artifacts remain preserved separately in the recovery workspace, unchanged. `historical/README.original.md` is an exact recovered historical document, whose results and commands are not validated by this release. Do not confuse its original pilot metrics with the new table above. Historical files and original manifests were not silently substituted for missing artifacts. New code and newly generated synthetic data are MIT licensed; see LICENSE.
