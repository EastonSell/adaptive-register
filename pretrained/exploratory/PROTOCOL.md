# Locally recorded pre-run protocol: exploratory instruction-label probe

This is a separately labeled follow-up to published pilot e906cb1517a8efbdc4da09002abf5c26c2e9a091. It is not a replacement analysis, independent preregistration, new held-out study, or retraining run. The 16 source records and their original outcomes have already been inspected. Source, exact prompts, scoring and input hashes are recorded before new inference and must pass independent review and a local hash freeze before running. Remote publication may follow execution while approval is pending; no public-pre-run claim is made. See AMENDMENT.md and the preserved initial protocol.

## What the saved evidence can support

Each trained arm has 25 trace rows: initial validation plus 24 optimizer updates. Validation **LM loss**, not answer accuracy, was recorded at steps 0/8/16/24. Only the validation-selected step-24 adapter is saved per arm; local pilot and packaged adapters are copies, not intermediate checkpoints. There are no per-update validation generations or accuracy measurements. We will not invent a validation-accuracy trajectory. The two auxiliary arms' original 48-row generation files are byte-identical, although their weights and losses differ.

The auxiliary head is not called during inference. That does not mean it cannot affect behavior: its training loss backpropagates into shared trainable LoRA parameters. This probe measures resulting LM behavior, not a fourth head class or head-classification accuracy.

## Question and fixed intervention

1. Does the untouched base reproduce its original generated text and strict format failures under the same pinned runtime and decoding settings?
2. For the same inspected facts, how do strict factual answers and format coverage behave when the requested instruction label changes to `concise`?

The existing `introductory`, `working`, and `technical` variants remain unchanged. The fourth prompt is an exact copy of the working prompt, replacing only `Requested register: working.` with `Requested register: concise.`. The system instruction still requires an answer beginning with `Value:`. `concise` is an unseen **instruction label**, not guaranteed to be a semantically distinct presentation mode; it may overlap working-style brevity. We do not invent a new exact target template or score whether prose is truly concise. All exact prompts are in `prompts.json`.

## Inference and evidence reuse

Use the same pinned base revision and verified model files, CPU float32, eager attention, two Torch threads, seed 11 and deterministic algorithms. Greedy generation uses the unchanged 24-new-token cap, tokenizer chat template, EOS/padding and decoding/stripping behavior. No temperature sampling, threshold loosening, output repair, optimizer, training, selection among checkpoints, or new model download is part of this probe.

Repeat the 48 original base prompts and run the fourth prompt on all 16 records for each of the four arms: 112 newly generated responses total. For the base's four-register metrics, use its freshly repeated original three outputs plus the new fourth output. For trained arms, combine the archived original-three outputs with the new fourth output. That evidence reuse is explicit, not a fresh four-register replication. All 144 trained-arm original generations were reproduced previously by saved-adapter reload, but are not generated again in this probe. Record original input/output/weight hashes and the source of each branch.

An existing output directory is refused. A 30-minute cooperative inference budget is checked between examples; it is not a hard wall-clock cap. Each completed response is appended to a fresh JSONL file, flushed and fsynced before generating the next response. Partial output is not a complete study. Base repeat text match is reported as a count out of 48; mismatches are preserved and disclosed, not silently discarded or retried until they disappear.

## Fixed scoring and reporting

Reuse the original `data.parse_answer` unchanged. Missing prefix, multiple numbers, unsupported digits, signed or decimal answers remain invalid. Every generation is retained. No post-hoc relaxed semantic scorer replaces it.

Prioritize record-level **all-four-registers-correct /16**, alongside all-three-registers-correct /16, strict correct response counts, format coverage, invalid-group counts, and answer changes across registers. Answer-change rate uses only groups whose every answer parses, and is undefined when no group is valid. Separately count groups whose answer-or-invalid status changes; invalid outputs must not be sold as stable factual answers. Report fourth-only strict correct and parsed counts /16. No arbitrary success gate, significance test, or checkpoint/model selection follows these exploratory results.

The 64 four-register responses per arm are correlated observations of only 16 source records. Neither this probe nor a repeated strict base score isolates semantics from formatting. Changes might reflect prompt wording, parsing, truncation or factual errors. No claim about human understanding, real-world language improvement, unseen facts, or broad presentation generalization is warranted. No further instruction labels or decode settings will be tried within this frozen probe.
