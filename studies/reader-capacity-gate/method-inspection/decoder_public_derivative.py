"""Narrow installed Llama + PEFT bridge. No model/checkpoint loader or study runner.

An already constructed CPU float32 Llama is required. This source can be tested
on a random tiny architecture without ever touching pretrained tensor files.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import torch
from torch import nn
from .accepted import core


def frozen_fingerprint(model, lora_parameters):
    """All non-LoRA parameter and all buffer bytes, including nonpersistent buffers."""
    digest = hashlib.sha256()
    for category, values in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, tensor in values:
            if category == 'parameter' and name in lora_parameters:
                continue
            metadata = [category, name, list(tensor.shape), str(tensor.dtype)]
            digest.update(json.dumps(metadata, separators=(',', ':')).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedLlama:
    model: nn.Module
    lora_parameters: dict

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False, return_dict=True)
        # No target pooling, shifts, detach, constructed tensor or model loss.
        return core.DecoderOutput(out.logits, out.hidden_states[-1])

    def greedy(self, prompt_ids, *, max_new_tokens=80, use_cache=True, eos_token_id=None):
        """One unpadded prompt; head-free argmax, include terminal EOS in returned IDs.

        Explicitly bounded. No training data/source access, sampling or tokenizer.
        """
        config = self.model.config
        if (not isinstance(prompt_ids, (tuple, list)) or not prompt_ids
            or not all(type(x) is int and 0 <= x < config.vocab_size for x in prompt_ids)):
            raise ValueError('prompt IDs must be nonempty in-vocabulary integers')
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 80:
            raise ValueError('generation cap must be an integer in 1..80')
        if type(use_cache) is not bool:
            raise ValueError('use_cache must be boolean')
        if len(prompt_ids) + max_new_tokens > config.max_position_embeddings:
            raise ValueError('prompt plus cap exceeds decoder context; truncation forbidden')
        eos = config.eos_token_id if eos_token_id is None else eos_token_id
        if type(eos) is not int or not 0 <= eos < config.vocab_size:
            raise ValueError('single in-vocabulary EOS integer required')
        from transformers.cache_utils import DynamicCache
        past = DynamicCache() if use_cache else None
        full = torch.tensor([prompt_ids], dtype=torch.long, device='cpu')
        generated = []
        training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for step in range(max_new_tokens):
                    current = full[:, -1:] if use_cache and step else full
                    out = self.model(input_ids=current,
                        attention_mask=torch.ones_like(full, dtype=torch.bool),
                        past_key_values=past, use_cache=use_cache,
                        output_hidden_states=False, return_dict=True)
                    token = int(out.logits[0, -1].argmax())
                    generated.append(token)
                    if token == eos:
                        break
                    full = torch.cat((full, torch.tensor([[token]], dtype=torch.long)), dim=1)
                    if use_cache:
                        past = out.past_key_values
        finally:
            self.model.train(training)
        return tuple(generated)


def prepare_lora(model, *, seed=11):
    """Freeze a fresh, already constructed Llama and inject exact q/v LoRA only.

    Changes the supplied model in place. Does not create/load a base model, read a
    checkpoint, download anything, or initialize the familiarity head.
    """
    from transformers import LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    if not isinstance(model, LlamaForCausalLM) or hasattr(model, 'peft_config'):
        raise ValueError('fresh unadapted LlamaForCausalLM required')
    if type(seed) is not int or seed < 0:
        raise ValueError('adapter seed must be a nonnegative integer')
    config = model.config
    if config._attn_implementation != 'eager' or config.pretraining_tp != 1:
        raise ValueError('eager Llama with pretraining_tp=1 required')
    if config.attention_dropout != 0.0:
        raise ValueError('attention dropout must be zero for this deterministic contract')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('CPU float32 parameters required')
    targets = {f'model.layers.{i}.self_attn.{projection}'
               for i in range(config.num_hidden_layers) for projection in ('q_proj', 'v_proj')}
    modules = dict(model.named_modules())
    actual = {name for name in modules if name.endswith(('.q_proj', '.v_proj'))}
    if actual != targets or any(type(modules[name]) is not nn.Linear for name in targets):
        raise ValueError('exact fresh q_proj/v_proj Linear set required')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapted = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', bias='none'))
    named = dict(adapted.named_parameters())
    expected = {f'base_model.model.{name}.lora_{matrix}.default.weight'
                for name in targets for matrix in ('A', 'B')}
    trainable = {n: p for n, p in named.items() if p.requires_grad}
    if set(trainable) != expected or any(p.grad is not None for p in named.values()):
        raise ValueError('unexpected trainable/gradient set after fresh LoRA injection')
    return PreparedLlama(adapted, trainable)
