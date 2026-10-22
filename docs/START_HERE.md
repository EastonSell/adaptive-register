# Start here: change the wording, keep the facts

**The question:** Can a model follow a requested presentation style without changing what it says is true?

Here, a **register** is an explicitly requested way of presenting an answer: `introductory`, `working`, or `technical`. It is not a guess about a reader's intelligence, age, or background. This repository tests mechanisms for separating content from presentation. It has **not demonstrated an auxiliary-head benefit or improved human comprehension**.

## A two-minute example

Suppose a check-in instruction must preserve three facts: **Bay 3**, **18:45**, and **printed ID**.

- Introductory: “Go to Bay 3 at 18:45. Bring printed ID.”
- Working: “Check-in: Bay 3, 18:45. Required: printed ID.”
- Technical: “Location=Bay 3; time=18:45; requirement=printed ID.”

“Check in this evening. Bring your ID” is shorter, but loses the exact place, time, and printed-copy requirement. Easier-looking wording is not automatically an equally informative answer. Conversely, repeating “Bay 9” in every version would be consistent but wrong.

![Illustrative content/presentation pipeline: the same three supplied facts enter a fixed formatter with three requested registers. All versions retain the literals; wrong supplied facts remain wrong. A separate training panel compares SFT, an auxiliary head, and a proposed stop-gradient control.](figures/content-presentation.svg)

The [tiny offline demonstration](examples/content_presentation.py) renders those fixed templates and checks literal presence. From the repository root, using Python 3.10+:

```sh
python docs/examples/content_presentation.py
python -m unittest discover -s docs/examples -p 'test_*.py' -v
```

This is **an illustration, not a new experiment**: no model, learning, semantic evaluator, or reader is involved. Its exact-string checklist can miss paraphrases and can accept negated text containing the same words. It teaches the difference between preserving supplied content and proving that content is correct or understood.

## What the controls test

A **control** is a simpler comparison that helps isolate why something changed.

- **Unconditioned content prediction:** do not let the style request affect the answer. A separate formatter can still change wording. If a more complicated method cannot beat this, complexity alone is not progress.
- **SFT (supervised fine-tuning):** train on examples of requested answers. This is the important “ordinary training” comparison, not an untouched model alone. The pretrained pilot uses LoRA: small trainable adapter matrices while the original base weights stay fixed.
- **SFT plus an auxiliary head:** add a small training-time classifier that predicts the requested register from shared representations. Its loss can update shared trainable parameters. Predicting a label correctly does not show that facts survived or readers understood. The published pretrained head is not used during generation.
- **Plus consistency:** penalize answer differences across register variants. This can encourage agreement, including agreement on a wrong answer.
- **Stop-gradient head:** let the classifier learn from representations, but block its loss from updating shared trainable parameters. This would help distinguish “the label is readable” from “the extra objective improves generation.” **This is an explanatory future control, not a completed arm in the published studies linked below.**

## What the measured results actually mean

Start with these three evidence levels, not with the largest percentage:

1. [Synthetic content/formatting separation](../milestone2/RESULTS.md): separated models never change the predicted integer across registers by construction. They do not beat the simpler unconditioned control on accuracy. Fixed template agreement is not learned language quality.
2. [Actual pretrained LoRA pilot](../pretrained/results/RESULTS.md): SFT, SFT + head, and SFT + head + consistency each scored **25/48 strict-correct responses**. Those prompts cover only **16 independent source records**, each repeated in three registers. The untouched base's 0/48 is strict parser/format failure, not proof that every response was semantically wrong. No extra-head advantage was observed in this one-seed pilot.
3. [SFT learning curves](../studies/sft-learning-curves/README.md): the later **46/48** is on **reused development validation**, not a fresh test, and uses no auxiliary head. A separate training repeat produced all scheduled numerical outputs through update 384, but its supervisor ended `incomplete/accounting_failure`. Numerical output and clean supervision are different claims.

Read **correctness**, **template matching**, and **answer consistency** separately. None is a human-comprehension, accessibility, or fairness measurement. Repeated variants and reused facts are not new independent evidence.

## Inspect first, reproduce at the right level

- Read the linked reports, raw generation links, split definitions, and limitations before rerunning anything.
- The [root README](../README.md#run-the-numpy-rebuild) gives the synthetic NumPy experiment. With its existing dependency available, `python tooling/run_tests.py` checks the NumPy and release-tool suites without requesting pretrained execution.
- The [pretrained setup](../pretrained/README.md) and [learning-curve verification guide](../studies/sft-learning-curves/REPRODUCE.md) distinguish retained-artifact checks, checkpoint replay, and new training. They have separate dependencies and local-model requirements. The two-minute example above needs none of them.

**Useful next questions:** Does an effect survive ordinary SFT and stop-gradient controls? Does it hold on genuinely new facts and independently authored text? Can readers answer factual questions more accurately, without losing uncertainty or conditions? Those require new, appropriately controlled evidence, not a relabeling of these examples.
