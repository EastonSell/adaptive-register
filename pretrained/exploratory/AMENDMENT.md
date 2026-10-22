# Prospective execution-timing and retention amendment

Recorded 2026-10-22 before any new probe generations. Independent source review passed the inference/scoring mechanics. GitHub publication approval may remain pending; the user requested continued local research. Therefore execution may begin after independent review and a local source/protocol/hash freeze, without waiting for remote publication. This is not a claim of public pre-run registration or publication.

The initial protocol is preserved in PROTOCOL.initial.md, SHA256 693de806132b649a8af5d282662d29820c4826841c2caed3c6bc664bc85ed2f4. The initial probe.py SHA256 was ff95ff3d1d169a0cec3c43a8b05c4dfa29904dd6ae8d4b031f994945a2d71eb9. The initial README also required publication first; that requirement is superseded.

In the same prospective correction, each completed response is written to a fresh JSONL file and flushed/fsynced before the next response begins. A regression verifies a completed row survives a later simulated timeout. Partial branches remain incomplete; their retained rows are not silently discarded. Hypotheses, facts, prompts, parser, decoding, model/adapter weights, scoring, and number of new responses are unchanged. No new probe outcomes were available when this amendment was made.
