# Exploratory instruction-label results

The untouched base's original format failure was exactly repeatable: **48/48 original generated texts matched**, with 0/48 parsed by the unchanged strict answer parser. Adding the unseen instruction label `concise` did not establish an auxiliary-head benefit. All three adapted arms remain correct across all four labels on **7/16 records**, with answer changes across labels on **4/16 records**.

## What was newly executed

This follow-up generated 112 responses: 48 original-prompt base repeats plus 16 `concise` responses from each of four arms. No training, model selection, new data, relaxed parsing or changed decoding was performed. The 16 facts had already been inspected. The original three outputs for trained arms are reused from the frozen selected-step artifacts, not newly repeated here. The base's four-label results use its fresh original-three repetition.

| Arm | All 3 labels correct | All 4 labels correct | Answer-change groups, 3 → 4 labels | Fourth-label strict correct | Fourth-label parsed |
|---|---:|---:|---:|---:|---:|
| Untouched base | 0/16 | 0/16 | undefined → undefined | 0/16 | 0/16 |
| SFT | 7/16 | 7/16 | 3/16 → 4/16 | 9/16 | 16/16 |
| SFT + auxiliary | 7/16 | 7/16 | 4/16 → 4/16 | 8/16 | 16/16 |
| SFT + auxiliary + consistency | 7/16 | 7/16 | 4/16 → 4/16 | 8/16 | 16/16 |

Across all four labels, SFT has 34/64 strict-correct responses; both auxiliary arms have 33/64. All 64 adapted responses per arm parse. The base has no parsed responses or fully valid groups, so its answer-change rate is undefined, not evidence of stable correct answers. These are only 16 correlated source records per arm, not 64 independent examples. The one-response SFT difference is descriptive, not a statistically established advantage.

`concise` is an unseen instruction label, not a demonstrated semantically distinct presentation mode. No brevity/readability score or new target template was used. Strict parsing still conflates answer formatting and factual correctness; a repeated base failure is not proof that every base response is semantically wrong. This probe does not isolate semantics or establish comprehension, new-fact generalization, or natural-language improvement.

## Timing and provenance

The initial source review passed inference/scoring mechanics. Before any new outcomes, execution timing was prospectively amended to allow an independently reviewed **local freeze** while GitHub publication approval remained pending. The initial protocol is preserved in `../exploratory/PROTOCOL.initial.md`; `../exploratory/AMENDMENT.md` describes the timing change and incremental-response retention. No public pre-run registration/publication claim is made.

Final pre-run protocol SHA256: `3f8a972015a57db148a128c9ac19df604f10a4655b5f026b3705c66e0f16d9f8`.

Frozen input manifest SHA256: `47bfefc87ea5a3d5c0ed6e88f50779a32c430117d957a7b1d2a5dd54e6f4b266`.

The original pilot's training source, protocol, adapters, data and results were unchanged. The run used the same immutable model revision, Python 3.12.14, Torch 2.5.1+cpu, Transformers 4.46.3, PEFT 0.13.2 and safetensors 0.4.5, with CPU float32, two threads and greedy 24-new-token decoding. The saved auxiliary head is not invoked during inference; its training gradients can still affect shared LoRA parameters. Inference inactivity must not be interpreted as proof that an auxiliary head cannot help.

The existing traces contain validation LM losses only at steps 0/8/16/24, not per-update validation accuracy; intermediate adapters were not saved. No validation-accuracy trajectory was reconstructed or invented.

## Verification

- All 112 newly generated responses are retained in JSONL and completed-branch JSON. Each completed row was flushed/fsynced before the next generation.
- JSONL and JSON rows match exactly. A separate raw-output recount matches the summary, checks every answer against the frozen record, and reports per-record predictions in `artifact-audit.json`. This recount was added after the run as an audit, not a new selection criterion.
- All 24 frozen input hashes still match after execution. No new labels, decode settings or success gates were tried.
- Eight exploratory unit tests and all 34 prior aggregate tests pass. These include interruption retention, strict record scoring and preservation of the original three prompt variants.

The result closes the previously missing untouched-base repetition and adds a limited wording-robustness observation. It does not resolve factual unreliability: only 7/16 records receive correct answers consistently across all four labels.
