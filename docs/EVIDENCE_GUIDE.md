# How to read the evidence

This project studies a useful but demanding goal: change an explanation for the reader without changing what it says is true. The published experiments test parts of that goal. They have **not demonstrated improved human comprehension or an advantage from an auxiliary register head**.

This guide covers the published evidence through [commit 66c97db](https://github.com/EastonSell/language-for-the-reader/commit/66c97db6c78fb62ef42739acc2ff67ba6ae84b80). It separates measured outcomes from ambitions and distinguishes verifying saved results from running a model again.

## Choose a starting point

- **Understand the idea:** read the [two-minute example](START_HERE.md). It illustrates that consistent wording can preserve a wrong fact.
- **See what failed:** read the [reader-specific rewrite examples below](#why-fluent-rewrites-still-fail).
- **Check a result without a model:** run the [saved-output verifier](#1-verify-the-reader-capacity-package-without-a-model).
- **Inspect the methods:** follow the [study map](#the-study-map), then open the relevant protocol, raw outputs and limitations.

## The study map

These are different experiments, not interchangeable scores on one benchmark.

Key terms: **SFT** means supervised fine-tuning on examples of desired answers. **LoRA** trains small adapter matrices while the base model weights remain frozen. **Strict correctness** in the structured pilot requires both the specified answer format and the correct digit. **Consistency** means agreeing across presentation variants, which can include agreeing on a wrong answer.

| Study | What it measures | Published result | What that result does not establish |
|---|---|---|---|
| [Synthetic content/format separation](../milestone2/RESULTS.md) | Integer-answer accuracy and whether the answer changes across explicitly requested registers | Separated width-32/46 models average 98.96%/99.31% accuracy with zero answer changes; the simpler unconditioned control reaches 99.57% | An accuracy advantage over the simpler control, learned prose quality, or reader understanding |
| [Pretrained LoRA pilot](../pretrained/results/RESULTS.md) | Strict structured-answer correctness, parser coverage, exact-template matching and cross-register answer changes | Ordinary SFT, SFT + auxiliary head, and SFT + head + consistency each score 25/48 strict-correct | An auxiliary-head advantage; 48 independent samples; or proof that the base model's unparseable answers are all semantically wrong |
| [Ordinary-SFT learning curves](../studies/sft-learning-curves/README.md) | How structured outputs change with more training exposure on reused development validation | Later output reaches 46/48; the separate repeat retains its failed supervisor status | A fresh test result, an auxiliary-objective effect, or clean monitored completion |
| [Reader-specific rewrite capacity gate](../studies/reader-capacity-gate/README.md) | AI-rated preservation of required source facts and support for concepts marked unfamiliar | 14/24 fluent truth-complete outputs versus 18 required; 3/24 all-concept-supported outputs versus 12 required | Passing the gate, human comprehension, or the benefit or failure of comparison arms that were not run |

The root README also links the earlier synthetic baseline and the exploratory fourth-label probe. The probe reuses already-inspected facts; do not treat it as a new independent test.

“Register” in the earlier studies means an explicit introductory, working or technical presentation request. In the capacity gate, each synthetic profile instead specifies known and unfamiliar concepts. Neither is an inference about a person's intelligence or demographics.

## What the latest capacity gate actually did

The question was practical: before comparing more elaborate training objectives, could one fixed ordinary-SFT attempt produce enough faithful, fluent, reader-specific rewrites?

The attempt adapted the pinned SmolLM2-135M-Instruct checkpoint with rank-4 query/value LoRA on CPU. It used one seed, 128 updates and 96 training scenarios. Each update presented one scenario's two complementary profile examples. All 96 scenarios appeared once, then 32 appeared again: **256 profile-example exposures**, not 128 distinct scenarios or 128 epochs.

Evaluation covered **12 development scenarios with two profiles each**, producing all 24 prescribed outputs once. Those paired outputs are correlated, not 24 independent reader observations. Generation was greedy, capped at 80 new tokens, with no output repair or favorable subset selection. Every output reached EOS before the cap. The development material was already exposed, including one scenario used in an earlier calibration.

Engineering review accepted the historical execution evidence, but the public package is narrower: it preserves the semantic audit, selected source snapshots and an engineering summary, not the complete training archive. See [methods](../studies/reader-capacity-gate/METHODS_AND_RESULTS.md) and [source scope](../studies/reader-capacity-gate/SOURCE_SCOPE.md).

## Read the measures separately

- **Fluency:** is the text intelligible connected English? This passed for 23/24 outputs.
- **Task following:** does it attempt the requested scenario rewrite? This passed for 24/24.
- **Truth-complete:** does it preserve every required proposition, including participants, quantities, negation, restrictions, timing and uncertainty? This passed for 14/24.
- **All unfamiliar concepts supported:** does it correctly explain or replace every concept marked unknown for that profile? This passed for 3/24.
- **Truth-safe:** is it truth-complete without adding unsupported scenario assertions? This passed for 12/24.

The historical gate required both separate floors: at least 18 fluent, task-following, truth-complete outputs and at least 12 all-concept-supported outputs. These were not joint-success thresholds, and the successful outputs did not have to be the same.

In fact, all three support successes, O006–O008, failed truth completeness. No output was both truth-safe and all-concept-supported. That stricter zero count is a useful diagnostic, **not a replacement gate invented after seeing the result**.

The unit matters. The final ratings preserve 64 of 74 individual propositions, but only 14 of 24 complete rewrites preserve every required proposition. Likewise, supporting 17 of 48 individual unfamiliar-concept items yields only three outputs with complete support. An item-level score can conceal a failure in an otherwise mostly correct explanation. See the [aggregate](../studies/reader-capacity-gate/results/aggregate.json).

## Why fluent rewrites still fail

These examples come from the retained outputs and final AI judgments, not new generations or new ratings.

### 1. A small word carries a large restriction

In [O006](../studies/reader-capacity-gate/outputs/O006.json), the source makes a signal conditional on being **only above** a threshold. The rewrite retains “above” but loses “only.” Both unfamiliar concepts receive acceptable explanations, yet the exclusive restriction is not preserved.

**Lesson:** a clear definition cannot compensate for a weakened condition. Check limiting words, not just the main nouns.

### 2. A correct-looking definition can name the wrong participant

In [O009](../studies/reader-capacity-gate/outputs/O009.json), the source places a solution to the left of a suspension. The rewrite describes the left-hand participant as a mixture with undissolved particles, which belongs to the suspension concept rather than the solution.

**Lesson:** definition accuracy and participant binding must both be checked. A glossary word match cannot establish that a relation survived.

### 3. Uncertainty must stay attached to the right claim

In [O013](../studies/reader-capacity-gate/outputs/O013.json), the source reports a click but leaves its cause untested. The rewrite makes the support or its clicking appear untested instead.

**Lesson:** retaining an uncertainty word is not enough. What is uncertain must also remain unchanged.

### 4. Perfect copying preserves facts but may not help this profile

[O003](../studies/reader-capacity-gate/outputs/O003.json) exactly copies its source. It preserves the required propositions but does not explain its stipulated unfamiliar terms.

**Lesson:** factual preservation is necessary, not sufficient, for the reader-specific task. Copying is not automatically task failure, but can fail concept support.

### 5. Preserving required facts does not license extra claims

[O017](../studies/reader-capacity-gate/outputs/O017.json) preserves the required propositions but specifies what an estimate is estimating beyond what the source explicitly states.

**Lesson:** truth-complete and truth-safe are different. Check additions as well as omissions.

Use the [source/profile packet](../studies/reader-capacity-gate/evaluation/source-aware-packet.json) alongside the [final ratings](../studies/reader-capacity-gate/adjudication/source-aware-final.json) to inspect each exact span and rationale. The [frozen rubric](../studies/reader-capacity-gate/evaluation/rubric.md) explains the scoring rules.

## Who judged the outputs?

The material, scoring ledger and judgments were AI-produced. A separate output-only AI answerer answered 74 frozen questions without seeing sources, expected answers, profiles, glossaries or reference rewrites. Later source-aware evaluation checked what the outputs actually preserved.

Two AI contexts produced initial source-aware ratings. Five categorical differences were explicitly reconciled against source/output evidence. The initial sets, final ratings and reconciliation remain distinct. Evaluator A's provisional all-supported count was four; the final count is three. Do not substitute the provisional count.

Separate AI contexts are not independent human validation and do not guarantee independent errors. The output-only phase's 69 answerable responses also do not mean 69 source-correct answers: answerability and source faithfulness are different outcomes. See [provenance](../studies/reader-capacity-gate/PROVENANCE.md).

## Reproduce the right claim

### 1. Verify the reader-capacity package without a model

From the repository root:

```sh
cd studies/reader-capacity-gate
python -B verify_package.py
```

This uses only the Python standard library. It checks the published file set and hashes, all 24 source/profile/output mappings, retained evidence spans, rating/reconciliation records and aggregation. Expected results include:

```text
public_package_verified: true
fluent_truth_complete: 14
unknown_support: 3
capacity_gate_passed: false
human_comprehension_measured: false
```

It does not install dependencies, load a model, call a service or rerate the language. A passing verifier establishes consistency and arithmetic of the saved package, **not the correctness of the AI interpretations or an end-to-end historical replay**. Follow the [complete verification guide](../studies/reader-capacity-gate/REPRODUCE.md).

### 2. Run the small explanation example

From the repository root:

```sh
python docs/examples/content_presentation.py
python -m unittest discover -s docs/examples -p 'test_*.py' -v
```

These fixed templates illustrate content/presentation separation. No model or reader is involved. Literal checks are not semantic evaluation.

### 3. Inspect or reproduce earlier experiments

- The [NumPy commands](../README.md#run-the-numpy-rebuild) run the synthetic experiments and tests. They need NumPy but no language-model download or credentials.
- The [pretrained guide](../pretrained/README.md) separates dependencies, pinned model assets, saved-adapter replay and new training. Replaying the released pilot's adapters previously matched all 144 trained-arm generations; that was checkpoint replay, not repeat training.
- The [learning-curve guide](../studies/sft-learning-curves/REPRODUCE.md) distinguishes package checks, evidence audits, checkpoint replay and new experiments. Its model-free tests still require torch and safetensors. Do not assume all model-free checks are standard-library-only.

For the capacity gate, complete training data, environment-specific wrappers and the full optimizer/gradient archive are not included. Its inspection-only source is not a portable model runner. Obtaining the pinned base model alone cannot recreate that historical run.

## The conclusion to carry forward

The published evidence makes a narrower but useful point: **presentation success, factual preservation, concept support, execution integrity and reader understanding must be assessed separately**.

The latest fixed capacity attempt failed its semantic gate despite accepted engineering evidence. It does not justify advancing to the planned comparison under that unchanged gate. It also cannot establish a general inability of the model, an effect of unrun objectives, or a benefit for human readers.

Any later experiment needs its own prospective protocol and evidence. A different method or new set of judgments must remain distinguishable from these retained results.
