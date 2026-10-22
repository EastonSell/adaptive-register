# Adaptive Register

A reproducible research foundation for adapting explanations to an **explicitly requested register** while keeping factual answers stable.

The experiment tests whether a learned auxiliary register head helps a shared response model follow a presentation request, and whether a counterfactual consistency objective prevents the answer from changing when only that request changes. The goal is to support a user's stated understanding, not to classify their intelligence or infer their background.

## What is here

- A NumPy neural model trained from scratch, with a joint answer/register head and an optional auxiliary head.
- An original synthetic record-retrieval dataset with record-disjoint splits and three register counterfactuals per question.
- Four ablations over five initialization seeds, raw metrics, all checkpoints, training curves as JSON, and representative responses.
- Numerical-gradient, leakage, learning, reproducibility, and checkpoint tests.
- Exported supervised-chat data and a resource-gated [open-model fine-tuning protocol](docs/open-model-protocol.md).
- [Primary-source research and claim matrix](docs/research.md), [human evaluation plan](docs/human-evaluation.md), and explicit [data/model provenance](docs/cards-and-provenance.md).

## Pilot finding

| Objective | Answer accuracy | Requested-register agreement | Answer changes across register requests |
|---|---:|---:|---:|
| No register input | 84.53 ± 2.20% | 33.33 ± 0.00% | 0.00 ± 0.00% |
| Conditioned joint objective | 84.48 ± 1.53% | 100.00 ± 0.00% | 11.56 ± 1.72% |
| + Auxiliary register head | 84.06 ± 1.41% | 100.00 ± 0.00% | 10.70 ± 2.81% |
| + Auxiliary head and consistency | 84.14 ± 1.29% | 100.00 ± 0.00% | 10.78 ± 2.93% |

Mean ± sample standard deviation over five model seeds, evaluated on one fixed test split. The auxiliary head offers **no register benefit in this easy control task**. The consistency penalty at the chosen weight does not eliminate factual drift. This negative result motivates a harder language-model evaluation rather than an improvement claim. See [full results](results/pilot/RESULTS.md), [raw scores](results/pilot/summary.json), and [interpretation](docs/results-analysis.md).

## Reproduce

Python 3.11+ and NumPy are sufficient. The recorded run used Python 3.12.14 and NumPy 2.3.5. No network, API key, model download, or GPU is used by these commands.

```bash
# In an environment that already has NumPy:
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m uat.experiment --config configs/pilot.json --out results/reproduction
python -m uat.export --splits results/reproduction/splits.json --out results/reproduction/chat
python scripts/compare_runs.py results/pilot/summary.json results/reproduction/summary.json
sha256sum -c MANIFEST.sha256
```

If preparing a new environment, `requirements.txt` pins the recorded NumPy version. Installation is a separate environment-management step, not performed by the experiment. Runtime on the recorded CPU is in `results/pilot/summary.json`; wall time depends on shared-host contention. The local experiment produces 20 small learned checkpoints. Public text-only checkouts may omit these optional binaries; the reproduction command regenerates them. Their reference hashes are in `CHECKPOINTS.sha256`.

## Method

For record/question features `x` and explicit register `r`, a 64-unit tanh encoder predicts a joint distribution `p(answer, register | x, r)`. Training minimizes joint cross-entropy, optionally plus register-head cross-entropy and mean-squared disagreement of answer marginals across the three counterfactual registers. The latter terms have weights 0.3 and 2.0. Decoding takes the separate answer and register marginal argmaxes. A finite template renders that predicted pair.

Data are split before creating register variants. Training shuffles intact groups. Checkpoints are chosen by validation joint cross-entropy, never test accuracy. The model learns which record slot answers a query; the answer is not supplied as an input feature. Oracle lookup is disclosed as a separate sanity control.

## Limitations

This release is a synthetic objective study, not a pretrained language-model fine-tune or a demonstration of improved human understanding. The model sees structured features, and the wording is templated. A perfect register label can coexist with a wrong answer; consistent answers can also be consistently wrong. Five initialization seeds on one generated dataset do not establish cross-domain generalization or statistical independence. No human study was conducted.

Only explicit `introductory`, `working`, and `technical` presentation requests are accepted. An absent request defaults to `working`. Demographic traits, dialect, education, occupation, and inferred intelligence are not model inputs. These constraints reduce one source of stereotyping but do not establish general fairness or accessibility.

## Next milestone

Run a license-checked, revision-pinned Qwen2.5-0.5B-Instruct LoRA smoke test using the exported chat data, then evaluate a separately authored source-grounded corpus and consenting users. The [protocol and resource gate](docs/open-model-protocol.md) specify objectives, masking, batching, baselines, evaluation, and stop conditions. The baseline does not allocate external resources. The next milestone can use suitable compute and isolated dependencies once its execution scope and resource budget are approved.

## License

Original code, documentation, synthetic records, and pilot checkpoints: [MIT](LICENSE). Dependencies and any future pretrained model retain their own licenses. See [provenance](docs/cards-and-provenance.md).
