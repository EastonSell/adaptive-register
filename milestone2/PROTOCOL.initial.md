# Content/register separation: preregistration

Written 2026-10-22 before new experiment execution. This is a synthetic structured-input study, not language-model fine-tuning or a comprehension study. Existing baseline results motivated the question; this is not blind discovery.

## Hypotheses and decision criteria

H1: preventing the requested register from entering the learned content predictor makes answer predictions invariant across register requests by construction. Test answer-change rate exactly zero, not a statistical improvement claim.
H2: separated content prediction will have higher held-out answer accuracy than the register-conditioned joint objective. Report each paired split/seed difference, mean, range, and count positive/tied/negative. Do not declare broad superiority from three overlapping synthetic splits.
H3: deterministically formatting a predicted answer can achieve requested-template agreement without establishing meaning preservation beyond the answer identifier or human understanding. Test that templates contain exactly the predicted answer integer and never replace it with the gold answer.

## Fixed design

Dataset seeds: 20261022, 20261023, 20261024. Model seeds: 11,22,33. Existing generator: enumerate 256 possible four-slot records, choose 160, partition 96/32/32 before expanding four query slots and three explicit registers. Record disjoint within each split; different splits reuse the finite universe. Queries and register variants are correlated. Report all nine runs per objective without treating them as nine independent datasets.

Objectives: original unconditioned joint, conditioned joint, auxiliary (0.3), auxiliary+consistency (0.3,2.0), separated content with hidden width32, separated content width46 for approximate active-parameter match to joint width32. Joint active parameters=1164 (1263 with auxiliary head); separated width32=804, width46=1154. Count unused auxiliary arrays separately from active parameters. Approximately matched parameters and equal update count are not matched wall time or equal optimization difficulty.

All use100 full-batch Adam updates, learning rate0.01, identical record splits, same seed set, and three repeated examples per record/query. Joint models receive record/query and explicit register except unconditioned register inputs are zero. Separated predictors receive only record/query20 features. Format selector receives only explicitly requested register and the predicted answer, never the source record or gold answer. A record lookup oracle is not used for model answers.

Checkpoint selection: original joint models use validation joint cross-entropy; separated models use validation answer cross-entropy. This difference is disclosed because objectives differ. Test data never select weights, hyperparameters or epochs. No tuning after observing new results in this milestone.

## Outcomes

Primary: held-out answer accuracy; fraction of record/query groups whose answers change across registers; fraction correct for all three registers. Secondary: learned register agreement for joint models; deterministic template-choice agreement for separated models (not comparable evidence of learning). Save every per-run score and paired differences. No significance tests or confidence intervals claimed from this small correlated design. Preserve new source, preregistration hash, and run config. Repeat execution and test deterministic scientific outputs.
