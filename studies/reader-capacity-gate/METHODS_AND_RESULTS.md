# Method and result

## Question and stopping rule

Could one fixed ordinary-SFT decoder attempt produce enough faithful, fluent reader-specific development rewrites, while correctly supporting concepts marked unfamiliar? This was a feasibility prerequisite, not the planned comparative efficacy study.

The procedure and floors were frozen before generation, but were not externally registered as a preregistration and are not a validated psychometric instrument. The denominator is all 24 expected outputs from 12 source scenarios, each with two complementary synthetic knowledge profiles. Missing, incomplete, uncertain or unresolved decisions cannot be removed from the denominator or credited as successes.

The unchanged historical gate required both:

- At least 18/24 fluent, task-following outputs with every required proposition preserved.
- Separately, at least 12/24 outputs accurately supporting every concept stipulated unknown for that profile.

The two numerators are not joint-success requirements. Truth-safe means truth-complete with no unsupported scenario assertion. Truth-safe plus all-concept support is a stricter reported descriptor, not a post hoc replacement threshold.

## Fixed model and exposure

The base was HuggingFaceTB/SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The attempt used CPU float32, eager attention, deterministic algorithms, two intra-op and one inter-op threads, and seed 11. No auxiliary head was constructed.

LoRA targeted q_proj/v_proj only: rank 4, alpha 8, dropout 0. Ordinary assistant-token-mean SFT used AdamW with learning rate 0.0005, weight decay 0, betas 0.9/0.999, epsilon 1e-8 and LoRA-only gradient clipping at 1.0. There were 128 updates, one training source and its two canonical complementary profile/reference variants per update, no gradient accumulation, and full training sequences at most 256 tokens.

The 96 training source IDs were sorted, shuffled once with Python 3.12 Random(11), then cycled without reshuffling. All 96 appeared once; the first 32 appeared again. This is 256 profile-example exposures, not 128 distinct sources or 128 epochs. The order clarification was made before outputs, not selected from results. The realized order is released in [method/training-order.json](method/training-order.json).

Each of the 24 assigned capacity-development prompts was generated once, in its fixed source/profile order, with greedy cached decoding and a maximum of 80 new tokens. EOS ID was 2. No early stopping, checkpoint selection, output repair, favorable subset selection or retuning was used. All outputs stopped with observed EOS before the cap; returned lengths were 33–51 token IDs including EOS. Termination is not semantic success.

The gate is deliberately exposed development data. In particular, kite_dock-01 and both profiles had already appeared in an earlier technical calibration. A fresh adapter/optimizer was used for this attempt, not calibration-trained state. No test record was used for training, generation or scientific evaluation in this attempt; however, the upstream alignment/data validator parsed combined corpus/token-case files, so this is not a claim that test bytes were physically unread.

## Engineering outcome is not the semantic outcome

An independent AI saved-evidence review accepted engineering custody: the original process exit was 0; all 128 update/optimizer calls and all 24 prescribed generations durably returned; mappings, decoded outputs and EOS handling reconciled. Same-version AdamW replay matched every retained consumed-gradient recurrence and final state. The reported engineering summary is [results/engineering-summary.json](results/engineering-summary.json).

These checks establish the reviewed execution evidence, not correctness of the generated language. Optimizer replay is conditional on recorded gradients and does not independently reconstruct derivatives or model activations. Periodic base/buffer fingerprints do not prove continuous immutability between observations. Sampled RSS and an in-process wall signal are not independent hard wall/RSS supervision. The complete authenticated engineering archive is not distributed here; the public verifier does not re-establish that historical engineering verdict.

## Semantic evaluation

A separate AI output-only answerer first received only each scoring text and its frozen proposition questions. It did not see sources, expected answers, profiles, glossaries or reference rewrites. Its answers were retained before source-aware review. This measures AI answerability, not human comprehension.

Source-aware judgments used the frozen scenario, proposition ledger, profile and glossary, not reference rewrite overlap. Every required proposition was classified preserved, omitted, contradicted or uncertain; every unfamiliar concept supported, inaccurate, absent/unglossed or uncertain. Passing decisions require exact output evidence and rationale. Fluency, task following and unsupported assertions were reviewed separately. Ordinary-language replacements can pass without retaining technical labels; a correct definition in the wrong participant slot cannot repair a changed scenario.

Two separate AI contexts completed initial source-aware ratings. Evaluator A also accessed the retained output-only record for later descriptive comparison; evaluator B excluded that record and evaluator A's judgments. Separate contexts are not independent human validation or a guarantee of statistically independent AI errors. There was no comparative arm blinding because only A was run.

All 74 proposition, 48 concept, 24 fluency and 24 task-following decisions were compared, plus all 24 unsupported-assertion reviews. Five categorical differences were resolved against exact source/output/glossary evidence, not averaged. An agreed final “uncertain” classification is still a semantic failure, not an unresolved disagreement. Both raw initial sets remain distinguishable from the final copy.

## Final counts

- Fluent and truth-complete: **14/24**, below **18** required.
- All unknown concepts supported: **3/24**, below **12** required.
- Truth-complete: 14/24. Truth-safe: 12/24. Truth-safe and all-supported: 0/24.
- Fluency: 23 pass, 1 fail. Task following: all 24 pass.
- Propositions: 64 preserved, 7 omitted, 2 contradicted, 1 uncertain; none unscored.
- Unknown concepts: 17 supported, 18 absent/unglossed, 2 inaccurate, 11 uncertain; none unscored.
- Six outputs contain unsupported assertions: O007, O009, O011, O013, O017 and O021.
- All 24 were adjudicated; zero missing/incomplete ratings or unresolved interrater disagreements.

All-unknown support passes were O006, O007 and O008. Their required propositions were not all preserved, demonstrating why separate support and truth counts matter. O003, O012 and O019 exactly copied the source: copying was not task failure, but did not explain stipulated unfamiliar terms.

The principal semantic failures were lost exclusive restrictions (O001/O002 “only after”; O005/O006 “only above”), missing measurement/time/location information (O004/O007/O008), swapped dissolved/undissolved mixture roles (O009/O011), and failure to preserve uncertainty specifically about a click's cause (O013). O017 and O021 preserved required core propositions but introduced unsupported specificity/properties and were therefore not truth-safe.

## Initial versus reconciled result

Evaluator A provisionally counted 14 fluent/truth-complete and 4 all-supported outputs; evaluator B counted 14 and 3. Reconciliation retained 14 and 3. Four disagreements refined diagnostic labels without changing an output-level pass; the fifth changed O014's gear explanation from supported to uncertain because a toothed wheel being driven does not explicitly convey passing motion to another part. [The full reconciliation](adjudication/reconciliation.json) includes all decisions, exact spans and rationales, not just these differences.

The retained output-only phase had 69 answerable, 3 uncertain and 2 unanswerable responses. Later ledger comparison found 64 consistent answers, five partial answers missing required scope, two missing-duration nonanswers, one appropriately uncertain entity answer later found in conflict with the source, one literal wrong-entity answer, and one causal-scope uncertainty. Answerability cannot restore missing restrictions or correct an output's participant substitution.

## Interpretation

The fixed semantic gate failed despite successful engineering execution. This attempt therefore does not justify proceeding to B/C/S efficacy comparisons under the unchanged plan. It estimates neither a benefit nor a lack of benefit for an auxiliary objective: those comparisons were not run here.

One small model, one fixed exposure budget, one seed, AI-authored synthetic material, correlated paired profiles, exposed development data and AI-only adjudication sharply limit the conclusion. This is not evidence of measured human comprehension, accessibility benefit, cognitive modeling, broad generalization or model-wide inability to adapt explanations. A later different design would be a new experiment with its own prospective protocol, not a repaired version of this failed attempt.
