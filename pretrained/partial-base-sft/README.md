# Partial pretrained pilot checkpoint

Untouched base and SFT control completed. Auxiliary-head and auxiliary-plus-consistency arms were still pending at this immutable snapshot. No final comparison conclusion.

Base: 0/48 outputs satisfy the frozen strict answer parser. This is format failure, not proof that every semantic answer is wrong; raw generations are included. SFT: 25/48 strict-correct, 48/48 parse coverage, 7/48 exact requested templates, 3/16 groups with different answers across registers. Validation selected update 24 from 0/8/16/24; 230,400 LoRA parameters were trained. This is actual pretrained adaptation on a tiny synthetic task, not demonstrated comprehension or general factual preservation.

One split, one seed, 16 held-out source records with three correlated register variants. Source/protocol checkpoint: abce2e6e335cbf59ec63095d968e18bc34c4a0a1. Adapter weight hash is recorded; this text checkpoint does not include the binary adapter. No data/model was changed based on held-out outputs. Runtime timings are shared-host observations, not speed benchmarks.
