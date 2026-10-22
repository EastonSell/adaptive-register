# Adaptive Register: recovery rebuild

A recovery research project testing explicit presentation-register requests, an auxiliary register head, and counterfactual answer consistency. The NumPy experiments below are newly written recovery implementations, not recovered original source. An additive [actual pretrained LoRA pilot](pretrained/results/RESULTS.md) now tests SmolLM2-135M-Instruct. Neither stage demonstrates improved comprehension or improvement of arbitrary models.

## Study guide

- [NumPy rebuild results](results/rebuild/summary.json): the synthetic experiment summarized below; its run and test commands are in the next section.
- [Content/formatting separation](milestone2/RESULTS.md): structural answer invariance without an accuracy gain over the simpler unconditioned control.
- [Pretrained LoRA pilot](pretrained/results/RESULTS.md) and [setup](pretrained/README.md): actual adaptation, with 25/48 strict-correct responses in each adapted arm and no observed auxiliary-head benefit.
- [Exploratory fourth-label probe](pretrained/exploratory-results/RESULTS.md): 112 new responses, including an exact untouched-base repeat; only 7/16 source records were correct across all four labels in each adapted arm. This reused already-inspected facts.
- [SFT learning curves](studies/sft-learning-curves/README.md): correctness reached 46/48 on reused development validation, not a new test set. The separate repeat produced the scheduled numerical outputs through update 384, while its supervisor ended incomplete with an accounting failure. [Verification and reproduction](studies/sft-learning-curves/REPRODUCE.md) distinguish retained-evidence checks from a new experiment.

## Run the NumPy rebuild

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

## NumPy rebuild results

Mean over three initialization seeds on one fixed split:

| Objective | Answer accuracy | Register agreement | Answer changes across registers | All registers correct |
|---|---:|---:|---:|---:|
| Unconditioned | 100.00% | 33.33% | 0.00% | 100.00% |
| Conditioned | 78.73% | 100.00% | 21.88% | 67.45% |
| Auxiliary | 75.09% | 100.00% | 23.96% | 62.76% |
| Auxiliary + consistency | 74.91% | 100.00% | 22.14% | 63.54% |

The extra head did not improve this toy task. Conditioning caused substantial factual degradation despite perfect register classification. Consistency did not solve it. This negative result is not evidence that the approach will work for natural language. Three seeds on 32 test records do not establish generalization or uncertainty across datasets. The four queries from each record are correlated.

## Verification and scope

The current NumPy suite has 17 passing tests (the original baseline had nine): record-disjoint splitting, counterfactual labels, determinism, finite-difference gradients including consistency, learning, checkpoint round trips, unconditioned invariance, renderer validation, and experiment artifacts. A second complete execution produced byte-identical summaries, splits, histories and all 12 checkpoints. These checks apply only to this new implementation; they do not reproduce the lost original implementation.

Presentation choices are `introductory`, `working`, `technical`; absent choice defaults to `working`. No demographic traits or inferred intelligence are used. No human comprehension, fairness or accessibility study was performed. The initial NumPy checkpoint did not contain pretrained fine-tuning. The later [pretrained milestone](pretrained/README.md) installs an isolated runtime and executes real LoRA adaptation, while retaining the earlier experiment unchanged.

## Recovery provenance

Original recovered artifacts remain preserved separately in the recovery workspace, unchanged. `historical/README.original.md` is an exact recovered historical document, whose results and commands are not validated by this release. Do not confuse its original pilot metrics with the new table above. Historical files and original manifests were not silently substituted for missing artifacts. New code and newly generated synthetic data are MIT licensed; see LICENSE.

## Content/formatting separation milestone

An additive [multi-split comparison](milestone2/RESULTS.md) tests keeping presentation choices outside content prediction. It achieves structural answer invariance, but does not beat the simpler unconditioned control on accuracy. This remains a synthetic experiment, not a pretrained fine-tune.

## Actual pretrained LoRA milestone

[Results and limitations](pretrained/results/RESULTS.md): all three adapted arms answer 25/48 held-out synthetic prompts correctly under a strict parser, with no observed accuracy benefit from the auxiliary head. The 48 prompts cover only 16 independent source records. This verifies actual fine-tuning execution, not reliable factual preservation or better language understanding.

## Aggregate release checks (post-run tooling)

With Python and NumPy installed, run `python tooling/run_tests.py` from the repository root. This runs the 17 NumPy tests and 8 standard-library tooling tests. It explicitly reports that the optional pretrained suite was not requested. To run all 34 tests (17 NumPy + 8 tooling + 9 pretrained), follow the isolated CPU setup and manifest-driven model fetch in [pretrained/README.md](pretrained/README.md), then run `python tooling/run_tests.py --pretrained --model-dir "$MODEL_DIR"`. Missing optional dependencies or model files fail clearly instead of silently skipping tests. These release tools were added after the frozen experiment; no training code, protocol, raw results, adapters or dependency pins were changed.
