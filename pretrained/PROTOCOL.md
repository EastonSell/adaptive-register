# Actual pretrained LoRA pilot: locally recorded pre-run protocol

Status: source/protocol checkpoint before training. This is an execution pilot on synthetic record queries, not a claim of better comprehension or generalized language quality. Previous NumPy experiments are separate and unchanged. No external preregistration or independent timestamp is implied.

## Model and environment

HuggingFaceTB/SmolLM2-135M-Instruct, immutable revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0 per its model card. Public safetensors weights and tokenizer only; `trust_remote_code=False`, local-files-only execution. No model-supplied Python or pickle weights are executed. Exact downloaded-file SHA256 hashes and installed runtime versions will accompany results. CPU float32, two PyTorch threads, no paid compute or credentials.

Official references:
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md
- https://huggingface.co/docs/transformers/v4.46.3/en/chat_templating
- https://huggingface.co/docs/peft/v0.13.0/en/package_reference/lora

## Data and interventions

Original synthetic data, generated using Python's random.Random seed 20261022: enumerate all four-slot records with values 0–3, shuffle, select 96 unique records and split 64 train / 16 validation / 16 test BEFORE variants. Each record receives one independently drawn query and three explicit register variants. No identical source record crosses splits. The outputs are templated natural-language targets containing the record's correct integer and the requested presentation. A single generation seed (11) is used: this smoke test cannot establish statistical significance.

Compare untouched pretrained model; SFT LoRA alone; SFT plus auxiliary register head (weight 0.1); SFT plus auxiliary head and answer-distribution consistency (weights 0.1 and 1.0). Every trained arm starts from the same immutable pretrained model and seeded rank-4 LoRA initialization on q_proj/v_proj, alpha 8, dropout 0.0. Base weights frozen. Auxiliary head is a trainable linear projection of the mean final-layer ASSISTANT target-token hidden states into three register labels. These teacher-forced targets reveal their template; auxiliary accuracy is therefore not evidence of understanding. No head is used to repair or rerank generated outputs.

Teacher-forced causal LM cross-entropy masks all prompt and padding tokens. Consistency is mean squared deviation of the four-number normalized distribution at the first answer digit's prediction position across three register counterfactuals of one source record. It does not guarantee correct content. Before proceeding, the actual tokenizer must give unique single-token digits 0–3 and unambiguous answer positions; otherwise stop and report an engineering protocol amendment before training.

## Fixed budget and selection

24 AdamW updates per trained arm, learning rate 0.0005, weight decay 0.0, gradient clipping 1.0. Each batch is exactly one source record with all three register requests; record order shuffled with seed 11 and cycled if needed. Equal updates/examples/LoRA rank, not identical parameter counts (auxiliary head adds parameters) or equal runtime. Validation at updates 0, 8, 16, 24 on all validation records, select lowest assistant-token-weighted validation LM cross-entropy, independent of auxiliary/consistency loss. Zero-update adapter is a candidate so training may correctly select no improvement. No test-guided selection or tuning. Stop on nonfinite loss, memory errors, >30 minutes at cooperative budget checks (not a hard wall-clock cap; a single load, update, or generation may overrun before the next check), or missing local inputs. Do not silently shrink data/change objectives to get a favorable result.

## Evaluation

Greedy generation, max_new_tokens 24, identical chat prompts and decoding for all arms. Save every held-out generated answer. Primary: strict extracted-answer correctness and generated answer-change rate across register requests. Parser requires exactly one numeric answer in 0–3 introduced by 'Value:' and no additional numbers; failures count incorrect. Also report coverage, invalid/mixed groups separately, all-registers-correct, and exact target-template match as a narrow style compliance proxy. Report generation token limit. Do not describe template matching as comprehension, accessibility or broad register quality. Compare pretrained, SFT and auxiliary arms on identical held-out examples. One small synthetic split and one seed cannot validate generalization; this is actual parameter-efficient fine-tuning feasibility, not an improvement guarantee.

## Durable outputs

Source/protocol first; then runtime/model hashes, train/validation traces, selected steps, trainable parameter counts, adapter safetensors hashes, raw test generations and summary. Adapter binaries retained locally pending publication review; public text can reproduce them from the pinned base model. Original model remains under Apache-2.0; new code and synthetic data remain MIT.
