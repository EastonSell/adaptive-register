# Experimental SmolLM2 LoRA adapters

Derived from HuggingFaceTB/SmolLM2-135M-Instruct, revision 12fd25f77366fa6b3b4b768ec3050bf629380bac, by Hugging Face / HuggingFaceTB. Original card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md

Base and derived weights are distributed under Apache-2.0; see ../APACHE-2.0.txt. Modification: rank-4 query/value LoRA adaptation on project-generated synthetic lookup prompts, 24 optimizer updates, seed 11. Auxiliary arms also train a three-way register head, saved separately. That head regularizes training; it is not needed or invoked for generation. No pretrained base weights are duplicated here.

These are experimental, not production-ready or safety-validated models. They fail nearly half of the held-out strict factual answers and do not demonstrate comprehension improvement. They must not be used to infer user intelligence or personal traits. Only explicit presentation requests are used.

Adapter configs replace the original local base-model path with the canonical model ID and pinned revision. Safetensors weight bytes are unchanged. See ../results/adapter-manifest.json for exact hashes and ../results/RESULTS.md for evaluation and limitations.
