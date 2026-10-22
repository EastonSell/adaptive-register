# Language for the Reader
## Separating factual content, requested presentation and measured reader understanding

Evidence snapshot: 22 October 2026. The long-term goal concerns useful explanations for readers. Existing measurements concern structured answers, requested registers and one small pretrained model.

## Abstract

This project asks whether an explanation can adapt its presentation while preserving factual content. Synthetic experiments compare joint answer/register prediction, an auxiliary register objective, answer-distribution consistency and structurally separated content prediction. Separation guarantees invariance of the predicted answer, but does not outperform a simpler unconditioned control on accuracy. A bounded SmolLM2-135M-Instruct LoRA pilot finds no strict-answer benefit from the auxiliary head: all adapted arms score 25/48 on 16 held-out records in three registers. Later ordinary-SFT learning curves reach 46/48 on reused development validation; their closure repeat retains complete scheduled numerical outputs alongside an incomplete supervision status. These studies identify evaluation and control requirements for a future reader-centered experiment; comprehension and natural-language semantic preservation remain unmeasured.

## Question and motivation

An introductory explanation and a technical explanation may differ in vocabulary, structure and detail while preserving the same claim, uncertainty and conditions. A reliable system must avoid using presentation requests as a reason to change the answer. It must also preserve enough information for the reader's task.

Three outcomes need separate measurement:

1. **Correctness:** is the content supported by the source?
2. **Stability and presentation:** does the factual answer remain stable across requested styles, and does the output follow that request?
3. **Understanding:** can the intended reader use the explanation correctly?

Consistency can preserve a wrong answer. A classifier can read a register label from a template without improving generation. Easier-looking prose may omit a necessary condition. These failure modes motivate the controls below.

## Related primary work

- [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858v2) establishes control-code conditioning as a language-generation approach. This project separately measures factual preservation under explicit control.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2) supplies the pretrained pilot's adaptation method: base weights stay frozen while low-rank updates are trained. The experiments test added objectives on top of ordinary SFT with LoRA.
- The [pinned SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) establishes model provenance and Apache-2.0 terms. Reader outcomes require their own evaluation beyond model-card benchmarks.

These are methodological precedents and provenance sources. This focused background is not a systematic review of explanation or comprehension research; such a review is needed before a human study.

## Hypotheses and controls

**Completed synthetic test:** withholding register from the content predictor should force identical answer predictions across register counterfactuals. The formatter receives the predicted integer and requested style, never the gold answer. The guarantee concerns invariance of the prediction; correctness is measured separately.

**Completed auxiliary-objective pilot:** a register head and consistency penalty might improve generated answers relative to ordinary SFT. Equal LoRA initialization, record order and update counts make SFT the central comparator. Auxiliary heads add parameters and work, so neither FLOPs nor total trainable parameters are exactly matched.

**Proposed next control:** a stop-gradient head would train the register classifier while preventing its loss from updating shared trainable parameters. Comparing it with an ordinary head and no-head SFT could separate label decodability from useful representation changes. This arm has not been completed in the published studies.

## Methods

The [NumPy rebuild](../README.md#run-the-numpy-rebuild) splits source records before expanding query/register variants. Its [separation extension](../milestone2/PREREGISTRATION.md) compares three overlapping splits and three initialization seeds per split. The unconditioned joint model, conditioned joint model, auxiliary objectives and separated predictors share a fixed update budget. Checkpoint objectives differ and are disclosed.

The [pretrained protocol](../pretrained/PROTOCOL.md) uses the immutable SmolLM2 revision, rank-4 query/value LoRA, seed 11 and 24 updates. Records split 64/16/16 before three variants are created. Validation LM loss selects among fixed candidates. Greedy generations are capped at 24 new tokens. A strict parser requires a single permitted digit introduced by `Value:`. The auxiliary head reads teacher-forced target states and is absent from generation.

The [learning-curve study](../studies/sft-learning-curves/README.md) uses ordinary SFT alone and previously inspected development validation. Exposure and optimization change together. Its later training repeat is distinct from resuming or obtaining a fresh test sample.

## Measured findings

- Across the synthetic separation study, width-32 and width-46 content predictors reached 98.96% and 99.31% mean accuracy with zero answer changes. The unconditioned control reached 99.57%; separation did not establish an accuracy advantage over that simpler control. [Results](../milestone2/RESULTS.md).
- SFT, auxiliary and auxiliary-plus-consistency each produced 25/48 strict-correct held-out generations, representing 16 source records. The untouched base produced no parseable responses under the frozen format, so its zero strict score cannot identify semantic correctness. No auxiliary-head benefit was observed. [Raw outputs, metrics and checkpoint replay](../pretrained/results/RESULTS.md).
- A [fourth-label probe](../pretrained/exploratory-results/RESULTS.md) reused already-inspected facts and is exploratory.
- Ordinary SFT reached 46/48 and 15/16 jointly correct records at update 384 on reused development validation. The separate repeat's worker exited zero, while its supervisor ended `incomplete/accounting_failure`. The final learning-curve result is not a clean supervised completion or held-out test result. [Evidence and failure record](../studies/sft-learning-curves/README.md).

## Reproducibility and limitations

[Start here](START_HERE.md) provides a model-free example. The [root commands](../README.md#run-the-numpy-rebuild), [pretrained setup](../pretrained/README.md) and [learning-curve guide](../studies/sft-learning-curves/REPRODUCE.md) distinguish software tests, retained-evidence checks, checkpoint replay and new training. Historical recovery material remains separate from the new implementation. The original manifest also covers an older README; later documentation changes require a new, separately scoped integrity record.

The finite record universe, synthetic templates, correlated variants, one pretrained seed and short decoding cap sharply limit inference. Parser compliance, exact-template matching and teacher-forced head accuracy each measure something narrower than language quality. No reader, accessibility, demographic fairness or comprehension outcome was measured. Presentation choices are explicit requests rather than inferred traits.

## Falsifiable next steps

1. Freeze independently authored facts and source-disjoint evaluation examples. Compare SFT, head, stop-gradient and consistency controls with fixed selection rules. A head that does not improve held-out factual preservation over SFT would fail the proposed benefit claim.
2. Add source-grounded scoring of omitted conditions, contradictions and uncertainty alongside strict structured checks. Retain every generation and score disagreement; do not replace the primary outcome after seeing failures.
3. After the content-preservation gate passes, design a consented reader study with randomized explanation conditions and predeclared factual questions. Measure accuracy, time and confidence separately. A comprehension claim requires improvement on the reader outcome without unacceptable loss of source content.

These steps are proposals. Additional corpus, training-core or harness candidates are not completed experiments or comprehension evidence.
