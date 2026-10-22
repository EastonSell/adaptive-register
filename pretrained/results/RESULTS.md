# Actual pretrained LoRA pilot: execution succeeds, extra-head benefit not shown

This is a completed adaptation of **HuggingFaceTB/SmolLM2-135M-Instruct**, not a randomly initialized substitute. All three trained arms learned the required answer format, but the auxiliary register head did not improve strict factual-answer accuracy over ordinary SFT. The model remains unreliable on this small lookup task.

## Frozen setup

The source and locally recorded pre-run protocol were published at `abce2e6e335cbf59ec63095d968e18bc34c4a0a1` before the run. Protocol SHA256 remains `3dabedfba9e74509675d0ae4aa7349a00736ae5a5353d8febec78090f1e67362`. This is not independent preregistration. The immutable partial base/SFT snapshot was published at `44f605e3540c38b1ca552984f41d301abebca118`; it remains unchanged.

Base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0. The base has 134,515,008 parameters. The model-file manifest records all downloaded file hashes. Execution used local safetensors, no remote model code, float32 CPU, two Torch threads, Python 3.12.14 and Torch 2.5.1+cpu. Complete dependency versions are in `../requirements.lock.txt`.

Records are split 64 training /16 validation /16 test before expanding three explicitly requested presentation registers. These are 16 independent held-out source records, **not 48 independent samples**. Each adapted arm uses the same seed 11, rank 4 query/value LoRA initialization, train-record order, 24 optimizer updates and validation candidates 0/8/16/24. Only 24 of the 64 training records are seen in this bounded smoke run. All arms selected update 24 by validation LM loss, never test outputs. Updates and LoRA parameter counts match; auxiliary arms add 1,731 head parameters and additional loss computation, so FLOPs/parameter budgets are not identical.

The head reads teacher-forced target representations and predicts the explicit register. Target templates reveal the label; this is not a measurement of human understanding. At inference the head is not used: prompts plus adapted LM generate freely, without copying or repairing answers from gold labels. The consistency objective compares four-digit answer distributions across register variants. It cannot guarantee correctness.

## Held-out results

| Arm | Strict correct | Parsed | Exact requested template | All 3 registers correct | Answer changes across registers |
|---|---:|---:|---:|---:|---:|
| Untouched base | 0/48 | 0/48 | 0/48 | 0/16 | undefined: no fully parsed groups |
| SFT | 25/48 | 48/48 | 7/48 | 7/16 | 3/16 |
| SFT + auxiliary | 25/48 | 48/48 | 7/48 | 7/16 | 4/16 |
| SFT + auxiliary + consistency | 25/48 | 48/48 | 7/48 | 7/16 | 4/16 |

**The base's zero score is strict format failure. It does not establish that all base responses are semantically wrong.** The frozen parser requires a `Value:` answer with one permitted digit, and generation is capped at 24 new tokens. Thus the base-to-adapter comparison conflates format compliance and factual correctness. Raw generations are retained for inspection; no post-hoc relaxed scoring replaces the primary metric. Exact-template matching is only compliance with synthetic target strings, not language quality or readability.

The extra objectives show no strict-accuracy or exact-template advantage in this one seed/split. Their observed answer-change count is one group higher than SFT, but this is not evidence of a statistically established general disadvantage. No human comprehension, accessibility, fairness, arbitrary-model transfer or natural-language improvement was tested. This small undertrained synthetic pilot should not be deployed as a factuality solution.

Validation LM loss was 3.071186 initially for all adapted arms and selected losses were 1.085430 SFT, 1.070012 auxiliary, 1.070424 auxiliary+consistency. Lower validation LM loss did not imply better held-out strict factual accuracy. Actual optimizer-updated LoRA B matrices contain nonzero values; tensor counts and norms are in `artifact-audit.json`.

## Verification and durable artifacts

- All 192 raw generation rows, exact split records, all three 25-row traces (initial validation plus 24 updates), and complete summary are included.
- Recomputing metrics from raw outputs exactly matches the saved summary; validation-selected steps match the trace minima.
- Packaged saved adapters were reloaded against the verified pinned base and **all 144/144 trained-arm generation rows matched exactly**. See `replay-verification.json`. This is saved-checkpoint replay, **not a repeat of training**; untouched-base generation was not repeated.
- Nine pretrained/replay tests and 17 prior NumPy tests pass. The training-mechanics unit test uses a tiny random model only to test gradients/frozen weights; the experiment uses the real pretrained model.
- Adapter configurations replace the original local base path with the canonical model ID and immutable revision. Safetensors bytes are unchanged. `adapter-manifest.json` records hashes for three adapters, two auxiliary heads and three configs. The heads are training regularizers, not required for generation.
- Base/SFT/auxiliary/consistency arm wall times were 70.18/98.05/97.51/91.00 seconds on a shared host. These are observations, not speed benchmarks. The 30-minute safety budget is cooperative, not a hard wall-clock cap.

The frozen protocol was not edited after observing outcomes. The replay utility, output-receipt regression test, this report and artifact packaging were added after the run. Results motivate better evaluation and controlled replication, not a claim that another objective will work.
