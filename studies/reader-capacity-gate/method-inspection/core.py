"""Architecture-neutral, training-only familiarity objective and exact controls.

No model loading, tokenization, generation, dataset optimization, or networking.
Decoder integration is a caller obligation documented in docs/INTEGRATION.md.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math
from collections.abc import Callable, Mapping, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ValidationError(ValueError):
    """An input violates the explicit objective/integration contract."""


class PoisonedControllerError(RuntimeError):
    """An incomplete optimizer-stage attempt permanently blocks this controller."""
    def __init__(self, failure_record):
        self.failure_record=copy.deepcopy(failure_record)
        super().__init__('controller is poisoned after an incomplete optimizer-stage attempt; retain state and stop, do not retry')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def finite_scalar(value: Tensor, label: str) -> Tensor:
    require(isinstance(value,Tensor) and value.ndim==0 and value.is_floating_point(), f'{label}: floating scalar required')
    require(bool(torch.isfinite(value)), f'nonfinite {label}; no optimizer update performed')
    return value


def form_objective(sft: Tensor, familiarity: Tensor | None, *, coefficient: float=0.1):
    """Validate reduced components and every actually formed scalar objective.

    The controller always uses 0.1. Other coefficients are useful only to test
    the helper's overflow checks; they do not alter the study prescription.
    """
    finite_scalar(sft,'SFT loss')
    require(isinstance(coefficient,(int,float)) and not isinstance(coefficient,bool) and math.isfinite(coefficient) and coefficient>0, 'objective coefficient must be positive finite')
    if familiarity is None:
        return None,finite_scalar(sft,'total objective')
    finite_scalar(familiarity,'familiarity loss')
    require(familiarity.device==sft.device and familiarity.dtype==sft.dtype, 'objective scalar device/dtype mismatch')
    weighted=finite_scalar(coefficient*familiarity,'weighted auxiliary objective')
    return weighted,finite_scalar(sft+weighted,'total objective')


def integer(value, label: str, minimum=0) -> int:
    require(type(value) is int and value >= minimum, f'{label}: expected integer >= {minimum}')
    return value


def text(value, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f'{label}: expected nonempty text')
    return value


def nonempty_list(value, label: str) -> list:
    require(isinstance(value, list) and bool(value), f'{label}: expected nonempty list')
    return value


@dataclass(frozen=True)
class AlignedExample:
    source_id: str
    profile_id: str
    input_ids: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    prompt_length: int
    span_positions: tuple[int, ...]
    known_labels: tuple[int, ...]


def align_tokenized_case(record: Mapping, profile: Mapping, case: Mapping,
                         *, max_sequence_length: int = 256) -> AlignedExample:
    """Validate the existing token-audit schema, deriving labels ONLY from profile.

    Offsets are Python character offsets in the retained rendered prompt. Leading
    whitespace within a subtoken is allowed; tokens crossing non-whitespace concept
    boundaries are rejected. This intentionally fails closed for merged boundaries.
    This validates retained structure, not token-ID-to-text authenticity: the caller
    must verify the pinned tokenizer/corpus/serialization hashes independently.
    """
    require(all(isinstance(x, Mapping) for x in (record, profile, case)), 'record/profile/case: expected mappings')
    integer(max_sequence_length, 'max_sequence_length', 2)
    sid = text(record.get('source_id'), 'source_id')
    pid = text(profile.get('profile_id'), 'profile_id')
    require(case.get('source_id') == sid and case.get('profile_id') == pid, 'token case identity mismatch')
    source = text(record.get('source'), 'source')
    prompt = text(case.get('prompt'), 'prompt')
    text(case.get('reference'), 'reference')
    require(case.get('errors') == [], 'retained token case has errors or lacks error status')
    start = prompt.find(source)
    require(start >= 0 and start == prompt.rfind(source), 'source must occur exactly once in rendered prompt')

    concepts = nonempty_list(record.get('concept_ids'), 'concept_ids')
    require(all(isinstance(c, str) and c.strip() for c in concepts) and len(set(concepts)) == len(concepts), 'concept_ids: nonempty unique strings required')
    known = nonempty_list(profile.get('known_concept_ids'), 'known_concept_ids')
    unknown = nonempty_list(profile.get('unknown_concept_ids'), 'unknown_concept_ids')
    require(all(isinstance(c, str) for c in known+unknown), 'profile concept IDs: strings required')
    require(len(set(known)) == len(known) and len(set(unknown)) == len(unknown), 'duplicate profile concept ID')
    require(not set(known)&set(unknown) and set(known)|set(unknown) == set(concepts), 'profile must partition source concept IDs')
    bits = profile.get('knowledge_bits')
    require(isinstance(bits, Mapping) and set(bits) == set(concepts), 'knowledge_bits: exact concept coverage required')
    require(all(type(v) is int and v in (0,1) for v in bits.values()), 'knowledge_bits: integer binary labels required')
    require(all(bits[c] == int(c in known) for c in concepts), 'knowledge_bits disagree with known/unknown partition')
    meanings = nonempty_list(profile.get('known_meanings'), 'known_meanings')
    require(all(isinstance(x, Mapping) and isinstance(x.get('concept_id'), str) and isinstance(x.get('meaning'), str) and x['meaning'].strip() for x in meanings), 'known_meanings: nonempty concept/meaning mappings required')
    require(len(meanings) == len(known) and {x['concept_id'] for x in meanings} == set(known), 'known_meanings: exact known concept coverage required')

    pi = nonempty_list(case.get('prompt_token_ids'), 'prompt_token_ids')
    ids = nonempty_list(case.get('input_plus_target_token_ids'), 'input_plus_target_token_ids')
    require(all(type(i) is int and i >= 0 for i in pi+ids), 'token IDs: nonnegative integers required')
    require(len(pi) < len(ids) <= max_sequence_length, 'sequence empty/overlength or no assistant target; truncation forbidden')
    require(ids[:len(pi)] == pi, 'prompt token IDs must be an exact full-sequence prefix')
    mask = case.get('assistant_token_mask')
    require(isinstance(mask, list) and all(type(v) is int and v in (0,1) for v in mask), 'assistant mask: integer 0/1 values required')
    require(mask == [0]*len(pi)+[1]*(len(ids)-len(pi)), 'assistant mask must select exactly the target suffix')
    for key, value in [('prompt_tokens',len(pi)), ('target_tokens',len(ids)-len(pi)), ('total_tokens',len(ids))]:
        require(type(case.get(key)) is int and case[key] == value, f'{key}: length metadata mismatch')
    offsets = case.get('prompt_token_offsets')
    require(isinstance(offsets, list) and len(offsets) == len(pi), 'prompt offset count mismatch')
    previous_start = -1
    for o in offsets:
        require(isinstance(o, (list,tuple)) and len(o) == 2 and all(type(x) is int for x in o), 'offset: integer pair required')
        a,b = o
        require(0 <= a <= b <= len(prompt), 'offset out of prompt bounds')
        if a != b:
            require(a >= previous_start, 'positive token offsets must have monotone starts')
            previous_start = a  # overlapping offsets are legal for byte-split Unicode
    spans = nonempty_list(record.get('concept_spans'), 'concept_spans')
    retained = nonempty_list(case.get('concept_token_spans'), 'concept_token_spans')
    require(len(retained) == len(spans), 'retained/source concept span count mismatch')
    intervals, final_positions, labels, seen_concepts = [], [], [], set()
    for span, item in zip(spans, retained):
        require(isinstance(span, Mapping) and isinstance(item, Mapping), 'span must be a mapping')
        cid = span.get('concept_id')
        require(isinstance(cid,str) and cid in concepts, 'span has unknown concept ID')
        a = integer(span.get('start'), 'span start')
        b = integer(span.get('end'), 'span end', 1)
        require(a < b <= len(source), 'span start/end out of source bounds')
        require(isinstance(span.get('text'),str) and bool(span['text'].strip()) and source[a:b] == span['text'], 'span text/source substring mismatch')
        require(not any(a < y and b > x for x,y in intervals), 'duplicate/overlapping concept spans forbidden')
        intervals.append((a,b)); seen_concepts.add(cid)
        require(all(item.get(k) == span[k] and type(item.get(k)) is type(span[k]) for k in ('concept_id','start','end')), 'retained/source span identity mismatch')
        left,right = start+a,start+b
        indices = [i for i,(x,y) in enumerate(offsets) if x < y and x < right and y > left]
        require(bool(indices), 'source concept span has no aligned subtokens')
        require(indices == list(range(indices[0],indices[-1]+1)), 'concept subtokens must be contiguous')
        require(item.get('token_indices') == indices and all(type(i) is int for i in item['token_indices']), 'retained token indices mismatch')
        require(type(item.get('final_subtoken')) is int and item['final_subtoken'] == indices[-1], 'retained final_subtoken mismatch')
        covered = [False]*(right-left)
        for i in indices:
            x,y = offsets[i]
            require(not prompt[x:min(y,left)].strip() and not prompt[max(x,right):y].strip(), 'subtoken crosses a non-whitespace concept boundary')
            for j in range(max(x,left),min(y,right)):
                covered[j-left] = True
        require(all(covered), 'subtoken offset coverage has a gap inside concept span')
        require(indices[-1] not in final_positions, 'multiple concepts share a final subtoken')
        final_positions.append(indices[-1]); labels.append(bits[cid])
    require(seen_concepts == set(concepts), 'source concept spans lack complete concept coverage')
    return AlignedExample(sid,pid,tuple(ids),tuple(mask),len(pi),tuple(final_positions),tuple(labels))


@dataclass(frozen=True)
class TrainingBatch:
    input_ids: Tensor
    attention_mask: Tensor
    assistant_mask: Tensor
    span_rows: Tensor
    span_positions: Tensor
    known_labels: Tensor
    case_ids: tuple[tuple[str,str], ...]

    def to(self, device) -> 'TrainingBatch':
        return TrainingBatch(*(getattr(self,k).to(device) for k in
                               ('input_ids','attention_mask','assistant_mask','span_rows','span_positions','known_labels')),
                             self.case_ids)


def validate_batch(batch: TrainingBatch) -> None:
    require(isinstance(batch, TrainingBatch), 'expected TrainingBatch')
    ids = batch.input_ids
    require(isinstance(ids,Tensor) and ids.dtype == torch.long and ids.ndim == 2 and ids.shape[0] > 0 and ids.shape[1] > 1, 'input_ids: nonempty [batch,time] int64 required')
    require(bool((ids >= 0).all()), 'negative token ID')
    b,t = ids.shape
    for name in ('attention_mask','assistant_mask'):
        m = getattr(batch,name)
        require(isinstance(m,Tensor) and m.dtype == torch.bool and m.shape == ids.shape and m.device == ids.device, f'{name}: bool tensor matching input IDs required')
    require(isinstance(batch.case_ids,tuple) and len(batch.case_ids)==b and all(isinstance(x,tuple) and len(x)==2 and all(isinstance(y,str) and y.strip() for y in x) for x in batch.case_ids), 'case_ids: source/profile string pairs required')
    require(len(set(batch.case_ids))==b, 'case_ids: unique source/profile identity per row required')
    lengths = batch.attention_mask.sum(1)
    require(bool((lengths > 1).all()), 'empty or one-token active sequence')
    positions = torch.arange(t,device=ids.device).unsqueeze(0)
    require(torch.equal(batch.attention_mask,positions < lengths[:,None]), 'attention_mask must be active prefix followed by right padding')
    targets = batch.assistant_mask.sum(1)
    require(bool((targets > 0).all()), 'every row needs assistant target tokens')
    prompts = lengths-targets
    require(bool((prompts > 0).all()), 'every row needs nonempty prompt')
    require(torch.equal(batch.assistant_mask,(positions >= prompts[:,None]) & (positions < lengths[:,None])), 'assistant_mask must select exactly active assistant suffix')
    for name in ('span_rows','span_positions'):
        x = getattr(batch,name)
        require(isinstance(x,Tensor) and x.dtype == torch.long and x.ndim == 1 and x.numel()>0 and x.device == ids.device, f'{name}: nonempty int64 vector on batch device required')
    r,p,y = batch.span_rows,batch.span_positions,batch.known_labels
    require(r.shape == p.shape and isinstance(y,Tensor) and y.shape == p.shape and y.is_floating_point() and y.device == ids.device, 'span indices/labels must be equal-length vectors')
    require(bool(((r>=0)&(r<b)).all()) and bool((p>=0).all()), 'span row/position out of bounds')
    require(bool((p < prompts[r]).all()), 'source span must lie strictly before assistant target')
    require(bool(((y==0)|(y==1)).all()) and bool(torch.isfinite(y).all()), 'known labels must be finite binary values')
    require(set(r.tolist()) == set(range(b)), 'every batch row needs annotated source spans')
    require(len(set(zip(r.tolist(),p.tolist()))) == r.numel(), 'duplicate span final position within row')


def collate_examples(examples: Sequence[AlignedExample], *, pad_token_id: int) -> TrainingBatch:
    integer(pad_token_id,'pad_token_id')
    require(bool(examples), 'cannot collate empty examples')
    require(all(isinstance(x,AlignedExample) for x in examples), 'expected AlignedExample inputs')
    for x in examples:
        require(bool(x.input_ids) and len(x.assistant_mask)==len(x.input_ids), 'example ID/mask length mismatch')
        require(all(type(i) is int and i>=0 for i in x.input_ids), 'example token IDs must be nonnegative integers')
        require(type(x.prompt_length) is int and 0<x.prompt_length<len(x.input_ids), 'example needs prompt and target')
        require(all(type(v) is int for v in x.assistant_mask) and x.assistant_mask==(0,)*x.prompt_length+(1,)*(len(x.input_ids)-x.prompt_length), 'example assistant mask mismatch')
        require(len(x.span_positions)==len(x.known_labels)>0, 'example span/label length mismatch')
        require(all(type(p) is int and 0<=p<x.prompt_length for p in x.span_positions) and all(type(y) is int and y in (0,1) for y in x.known_labels), 'example span/label invalid')
    b,t = len(examples),max(len(x.input_ids) for x in examples)
    ids = torch.full((b,t),pad_token_id,dtype=torch.long)
    attention,assistant = torch.zeros((b,t),dtype=torch.bool),torch.zeros((b,t),dtype=torch.bool)
    rows,positions,labels=[],[],[]
    for i,x in enumerate(examples):
        ids[i,:len(x.input_ids)]=torch.tensor(x.input_ids)
        attention[i,:len(x.input_ids)]=True
        assistant[i,:len(x.input_ids)]=torch.tensor(x.assistant_mask,dtype=torch.bool)
        rows.extend([i]*len(x.span_positions)); positions.extend(x.span_positions); labels.extend(x.known_labels)
    batch = TrainingBatch(ids,attention,assistant,torch.tensor(rows,dtype=torch.long),torch.tensor(positions,dtype=torch.long),torch.tensor(labels,dtype=torch.float32),tuple((x.source_id,x.profile_id) for x in examples))
    validate_batch(batch)
    return batch


def assistant_token_ce(logits: Tensor, batch: TrainingBatch) -> Tensor:
    validate_batch(batch)
    require(isinstance(logits,Tensor) and logits.ndim==3 and logits.shape[:2]==batch.input_ids.shape and logits.shape[2]>1 and logits.is_floating_point() and logits.device==batch.input_ids.device, 'logits: floating [batch,time,vocab] tensor on batch device required')
    require(bool(torch.isfinite(logits).all()), 'nonfinite logits')
    require(bool((batch.input_ids < logits.shape[-1]).all()), 'token ID outside decoder vocabulary')
    mask=batch.assistant_mask[:,1:]
    return finite_scalar(F.cross_entropy(logits[:,:-1][mask],batch.input_ids[:,1:][mask],reduction='mean'),'SFT loss')


class FamiliarityHead(nn.Module):
    """Scalar linear head initialized with a private CPU torch.Generator stream."""
    def __init__(self, hidden_size: int, *, seed: int):
        super().__init__()
        integer(hidden_size,'hidden_size',1); integer(seed,'head seed')
        rng=torch.Generator(device='cpu'); rng.manual_seed(seed)
        scale=1/math.sqrt(hidden_size)
        self.weight=nn.Parameter(torch.empty(hidden_size).uniform_(-scale,scale,generator=rng))
        self.bias=nn.Parameter(torch.empty(()).uniform_(-scale,scale,generator=rng))

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden @ self.weight + self.bias


def familiarity_bce(hidden: Tensor, batch: TrainingBatch, head: FamiliarityHead,
                    *, stop_gradient: bool) -> Tensor:
    validate_batch(batch)
    require(type(stop_gradient) is bool, 'stop_gradient must be boolean')
    require(isinstance(hidden,Tensor) and hidden.ndim==3 and hidden.shape[:2]==batch.input_ids.shape and hidden.shape[2]==head.weight.numel() and hidden.is_floating_point() and hidden.device==batch.input_ids.device, 'hidden: floating [batch,time,hidden_size] tensor on batch device required')
    require(head.weight.device==hidden.device and head.weight.dtype==hidden.dtype, 'head/hidden device or dtype mismatch')
    require(bool(torch.isfinite(hidden).all()), 'nonfinite hidden states')
    selected=hidden[batch.span_rows,batch.span_positions]
    if stop_gradient:
        selected=selected.detach()
    scores=head(selected)
    require(bool(torch.isfinite(scores).all()), 'nonfinite familiarity logits')
    return finite_scalar(F.binary_cross_entropy_with_logits(scores,batch.known_labels.to(scores.dtype),reduction='mean'),'familiarity loss')


@dataclass(frozen=True)
class DecoderOutput:
    logits: Tensor
    last_hidden_state: Tensor


def grad_norm(grads) -> float:
    values=[g.detach().double().square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def component_gradients(loss: Tensor, params: tuple[nn.Parameter,...]):
    require(loss.requires_grad, 'objective has no differentiable graph')
    return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)


def differentiated_norm(loss: Tensor, params: tuple[nn.Parameter,...]) -> float:
    return grad_norm(component_gradients(loss,params))


class ObjectiveController:
    """One matched forward/update, distinct AdamW optimizers, LoRA-only clipping.

    The exact explicit mapping must equal the generator's entire trainable set.
    No base parameter is silently frozen. Callers must prepare/freeze LoRA first.
    A uses SFT. B adds 0.1 BCE through hidden states. C detaches them for BCE.
    Diagnostics are measured per step on the same graph without extra forwards.
    """
    auxiliary_coefficient=0.1

    def __init__(self, generator: nn.Module, lora_parameters: Mapping[str,nn.Parameter],
                 *, arm: str, hidden_size: int, head_seed: int,
                 learning_rate: float=0.0005, max_grad_norm: float=1.0):
        require(arm in ('A','B','C'), 'arm must be A, B, or C')
        require(isinstance(generator,nn.Module), 'generator must be a torch module')
        require(isinstance(lora_parameters,Mapping) and bool(lora_parameters), 'explicit nonempty LoRA parameter map required')
        named=dict(generator.named_parameters())
        require(set(lora_parameters)=={n for n,p in named.items() if p.requires_grad}, 'explicit LoRA map must equal the entire trainable generator parameter set')
        require(all(named[n] is p for n,p in lora_parameters.items()), 'LoRA parameter names must identify generator-owned tensors')
        require(len({id(p) for p in lora_parameters.values()})==len(lora_parameters), 'duplicate/aliased LoRA parameters')
        require(all(p.is_floating_point() for p in lora_parameters.values()), 'LoRA tensors must be floating point')
        require(isinstance(learning_rate,(int,float)) and not isinstance(learning_rate,bool) and math.isfinite(learning_rate) and learning_rate>0, 'learning_rate must be positive finite')
        require(isinstance(max_grad_norm,(int,float)) and not isinstance(max_grad_norm,bool) and math.isfinite(max_grad_norm) and max_grad_norm>0, 'max_grad_norm must be positive finite')
        self.generator=generator; self.arm=arm; self.max_grad_norm=float(max_grad_norm)
        self._status='ready'; self._failure=None
        self._completed_updates=0; self._attempted_steps=0
        self.parameter_identities={n:id(p) for n,p in named.items()}
        self.lora_names=tuple(sorted(lora_parameters)); self.lora=tuple(lora_parameters[n] for n in self.lora_names)
        self.base=tuple(p for n,p in named.items() if n not in lora_parameters)
        require(bool(self.base), 'generator must contain frozen base parameters')
        require(len({(p.device,p.dtype) for p in self.lora})==1, 'LoRA parameters need a single device/dtype')
        self.head=None if arm=='A' else FamiliarityHead(hidden_size,seed=head_seed).to(device=self.lora[0].device,dtype=self.lora[0].dtype)
        self.generator_optimizer=torch.optim.AdamW(self.lora,lr=learning_rate,weight_decay=0.0)
        self.head_optimizer=None if self.head is None else torch.optim.AdamW(self.head.parameters(),lr=learning_rate,weight_decay=0.0)
        self._check_parameters()

    @property
    def status(self):
        return self._status

    @property
    def completed_updates(self):
        return self._completed_updates

    @property
    def failure_record(self):
        return copy.deepcopy(self._failure)

    def state_record(self):
        """JSON-safe safety status, not a full tensor checkpoint or resume API."""
        return dict(schema_version=1,status=self._status,arm=self.arm,
                    attempted_steps=self._attempted_steps,completed_updates=self._completed_updates,
                    failure=self.failure_record)

    def _check_parameters(self):
        require(type(self.auxiliary_coefficient) is float and self.auxiliary_coefficient==0.1, 'controller auxiliary coefficient must remain 0.1')
        named=dict(self.generator.named_parameters())
        require({n:id(p) for n,p in named.items()}==self.parameter_identities, 'generator parameter identities changed after controller creation')
        require({n for n,p in named.items() if p.requires_grad}==set(self.lora_names) and all(named.get(n) is p for n,p in zip(self.lora_names,self.lora)), 'generator trainable parameters changed after controller creation')
        require(all(not p.requires_grad and p.grad is None for p in self.base), 'base parameters must remain frozen and gradient-free')
        expected={id(p) for p in self.lora}
        actual=[p for g in self.generator_optimizer.param_groups for p in g['params']]
        require(len(actual)==len(expected) and {id(p) for p in actual}==expected, 'generator optimizer parameter set changed')
        if self.head_optimizer is not None:
            hp=[p for g in self.head_optimizer.param_groups for p in g['params']]
            require(len(hp)==len(tuple(self.head.parameters())) and {id(p) for p in hp}=={id(p) for p in self.head.parameters()} and not expected&{id(p) for p in hp}, 'head optimizer parameter set changed or overlaps generator')

    def step(self, batch: TrainingBatch,
             forward: Callable[[Tensor,Tensor],DecoderOutput]) -> dict:
        if self._status!='ready':
            raise PoisonedControllerError(self._failure)
        self._check_parameters(); validate_batch(batch)
        self._attempted_steps+=1
        exposure=dict(case_ids=[list(x) for x in batch.case_ids],
                      assistant_tokens=int(batch.assistant_mask.sum()),annotated_spans=batch.span_rows.numel())
        output=forward(batch.input_ids,batch.attention_mask)
        require(isinstance(output,DecoderOutput), 'forward must return DecoderOutput')
        sft=assistant_token_ce(output.logits,batch)
        fam=None if self.head is None else familiarity_bce(output.last_hidden_state,batch,self.head,stop_gradient=self.arm=='C')
        # Finite inputs and finite gradients do NOT imply a finite reduction or
        # formed scalar objective. Check all of them before any autograd work.
        weighted,total=form_objective(sft,fam,coefficient=self.auxiliary_coefficient)
        require(sft.requires_grad, 'SFT loss is structurally disconnected from selected LoRA parameters')
        sft_grads=component_gradients(sft,self.lora)
        sft_connected=sum(g is not None for g in sft_grads)
        require(sft_connected>0, 'SFT loss is structurally disconnected from selected LoRA parameters')
        aux_grads=() if fam is None else component_gradients(fam,self.lora)
        aux_connected=sum(g is not None for g in aux_grads)
        if self.arm=='B':
            require(aux_connected>0, 'B auxiliary loss is structurally disconnected from selected LoRA parameters')
        if self.arm=='C':
            require(aux_connected==0, 'C auxiliary loss unexpectedly connects to selected LoRA parameters')
        sft_norm=grad_norm(sft_grads);aux_norm=grad_norm(aux_grads)
        head_norm=0.0 if fam is None else differentiated_norm(fam,tuple(self.head.parameters()))
        # A/C generator backward is the same SFT path. Connected zero derivatives
        # are valid; unused parameters remain None rather than becoming fake zeros.
        self.generator_optimizer.zero_grad(set_to_none=True)
        if self.head_optimizer is not None:
            self.head_optimizer.zero_grad(set_to_none=True)
        sft.backward(retain_graph=self.arm=='B')
        if weighted is not None:
            weighted.backward()
        self._check_parameters()
        all_grads=[p.grad for p in self.lora]+([] if self.head is None else [p.grad for p in self.head.parameters()])
        require(all(g is None or bool(torch.isfinite(g).all()) for g in all_grads), 'nonfinite gradients; no optimizer update performed')
        before=grad_norm(p.grad for p in self.lora)
        torch.nn.utils.clip_grad_norm_(self.lora,self.max_grad_norm,error_if_nonfinite=True)
        after=grad_norm(p.grad for p in self.lora)
        measured_head=0.0 if self.head is None else grad_norm(p.grad for p in self.head.parameters())
        metrics=dict(arm=self.arm,sft_loss=float(sft.detach()),
                     familiarity_loss=None if fam is None else float(fam.detach()),
                     weighted_auxiliary_loss=None if weighted is None else float(weighted.detach()),
                     total_loss=float(total.detach()),auxiliary_coefficient=self.auxiliary_coefficient,
                     assistant_tokens=exposure['assistant_tokens'],annotated_spans=exposure['annotated_spans'],
                     sft_lora_grad_l2=sft_norm,familiarity_lora_grad_l2=aux_norm,
                     sft_lora_connected_tensors=sft_connected,familiarity_lora_connected_tensors=aux_connected,
                     lora_tensor_count=len(self.lora),
                     weighted_familiarity_lora_grad_l2=self.auxiliary_coefficient*aux_norm,
                     familiarity_head_grad_l2=head_norm,weighted_familiarity_head_grad_l2=measured_head,
                     generator_grad_l2_before_clip=before,generator_grad_l2_after_clip=after,
                     generator_clip=self.max_grad_norm,base_gradient_count=sum(p.grad is not None for p in self.base))
        stage='generator_optimizer';generator_outcome='uncertain_may_have_mutated';head_outcome='not_called'
        try:
            self.generator_optimizer.step()
            generator_outcome='returned'
            if self.head_optimizer is not None:
                stage='head_optimizer';head_outcome='uncertain_may_have_mutated'
                self.head_optimizer.step()
                head_outcome='returned'
            stage='post_update_validation'
            self._check_parameters()
        except BaseException as exc:
            # An optimizer may throw before OR after partial mutation. We cannot
            # infer its outcome from the exception and never pretend to roll back.
            self._status='poisoned'
            try:message=str(exc)
            except BaseException:message='<exception message unavailable>'
            self._failure=dict(schema_version=1,controller_state='poisoned',stage=stage,
                               attempt_index=self._attempted_steps,intended_update_index=self._completed_updates+1,
                               completed_updates_before_failure=self._completed_updates,
                               generator_optimizer_outcome=generator_outcome,head_optimizer_outcome=head_outcome,
                               exception=dict(type=type(exc).__name__,message=message),exposure=exposure,
                               pre_optimizer_metrics=metrics,
                               outcome='incomplete_optimizer_stage; retain tensors, optimizer states and gradients; stop without retry')
            raise
        self._completed_updates+=1
        return dict(metrics,completed_update_index=self._completed_updates)
