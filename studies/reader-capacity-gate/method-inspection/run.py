"""Fixed A-only orchestration. No model loader, resume or other arms."""
import copy
import torch
from .accepted import core,decoder
from .bootstrap import evaluation_cases
from .custody import tree_sha,file_sha
from .schedule import check_inputs,need


def state_payload(prepared,controller):
    return {'adapter':{n:p.detach().clone() for n,p in prepared.lora_parameters.items()},
            'optimizer':copy.deepcopy(controller.generator_optimizer.state_dict()),
            'gradients':{n:None if p.grad is None else p.grad.detach().clone() for n,p in prepared.lora_parameters.items()},
            'controller':controller.state_record(),'head':None,'head_optimizer':None}


def examples(row):
    result=[]
    for raw in row['examples']:
        value=dict(raw)
        for key in ('input_ids','assistant_mask','span_positions','known_labels'):value[key]=tuple(value[key])
        result.append(core.AlignedExample(**value))
    return tuple(result)


def execute(run,document,model_factory,decode,*,final_check,controller_factory=core.ObjectiveController):
    counts=dict(model_load_calls=0,forward_begins=0,forward_returns=0,update_attempts=0,completed_updates=0,
                optimizer_attempts=0,optimizer_returns=0,generation_calls=0,generation_returns=0,
                generated_candidates=0,generated_ids_returned=0)
    context={'phase':'prepare','update':None,'generation_id':None}
    prepared=controller=original_forward=original_step=None;opened=False;failure_snapshot=None
    try:
        schedule=check_inputs(document);cases=evaluation_cases(document)
        need(run.admission is not None,'frozen admission required before execution')
        run.save_json('inputs.json',document)
        need(file_sha(run.root/'inputs.json')==run.admission['expected_inputs_sha256'],'retained input identity differs from admission')
        train={r['source_id']:r for r in document['train']}
        run.check();counts['model_load_calls']+=1;run.event('model_load_begin',call=1)
        model=model_factory();run.event('model_load_returned',call=1);run.check()
        prepared=decoder.prepare_lora(model,seed=11)
        controller=controller_factory(prepared.model,prepared.lora_parameters,arm='A',hidden_size=prepared.model.config.hidden_size,head_seed=17)
        need(controller.head is None and controller.head_optimizer is None,'A must remain head-free')
        base=decoder.frozen_fingerprint(prepared.model,prepared.lora_parameters)
        run.event('decoder_prepared',seed=11,base_fingerprint=base,lora_names=sorted(prepared.lora_parameters),
                  lora_parameters=sum(p.numel() for p in prepared.lora_parameters.values()),
                  hidden_size=prepared.model.config.hidden_size,num_hidden_layers=prepared.model.config.num_hidden_layers,
                  eos_token_id=prepared.model.config.eos_token_id,vocab_size=prepared.model.config.vocab_size,
                  num_attention_heads=prepared.model.config.num_attention_heads,num_key_value_heads=prepared.model.config.num_key_value_heads)
        run.save_payload('A-initial.pt',state_payload(prepared,controller))
        original_forward=prepared.model.forward
        def tracked_forward(*args,**kwargs):
            run.check();need(not args,'explicit named inputs required')
            counts['forward_begins']+=1;call=counts['forward_begins']
            need(call<=2048,'forward operation cap exceeded')
            run.event('forward_begin',call=call,**context,input_ids=kwargs['input_ids'].detach().cpu().tolist(),
                      attention_mask=kwargs['attention_mask'].detach().cpu().tolist(),use_cache=bool(kwargs.get('use_cache',False)),model_training=prepared.model.training)
            try:out=original_forward(**kwargs)
            except BaseException as exc:
                run.event('forward_exception',call=call,**context,exception_type=type(exc).__name__);raise
            candidate=None
            if context['phase']=='generation':candidate=int(out.logits[0,-1].argmax());counts['generated_candidates']+=1
            counts['forward_returns']+=1
            run.event('forward_returned',call=call,**context,logits_shape=list(out.logits.shape),candidate_token_id=candidate)
            run.check();return out
        prepared.model.forward=tracked_forward
        original_step=controller.generator_optimizer.step
        def tracked_step(*args,**kwargs):
            counts['optimizer_attempts']+=1
            # Retain consumed gradients BEFORE the possibly partially mutating call.
            gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in prepared.lora_parameters.items()}
            identity=run.save_payload(f'A-consumed-{context["update"]:03}.pt',{'update':context['update'],'gradients':gradients})
            run.event('optimizer_begin',update=context['update'],gradient_file=identity['name'])
            try:result=original_step(*args,**kwargs)
            except BaseException as exc:
                run.event('optimizer_exception',update=context['update'],exception_type=type(exc).__name__,outcome='uncertain_may_have_mutated');raise
            counts['optimizer_returns']+=1
            run.event('optimizer_returned',update=context['update'],gradient_file=identity['name'],
                      adapter_sha256=tree_sha({n:p.detach() for n,p in prepared.lora_parameters.items()}),
                      optimizer_sha256=tree_sha(controller.generator_optimizer.state_dict()))
            run.check();return result
        controller.generator_optimizer.step=tracked_step
        prepared.model.train()
        for update,sid in enumerate(schedule,1):
            context.update(phase='update',update=update,generation_id=None);opened=True
            batch=core.collate_examples(examples(train[sid]),pad_token_id=2)
            counts['update_attempts']+=1
            run.event('update_begin',arm='A',update=update,source_id=sid,case_ids=batch.case_ids,
                      controller=controller.state_record(),assistant_tokens=int(batch.assistant_mask.sum()))
            run.check();metrics=controller.step(batch,prepared.forward);opened=False
            counts['completed_updates']+=1
            need(metrics['base_gradient_count']==0 and metrics['familiarity_loss'] is None,'A-only objective/base-gradient invariant failed')
            run.event('update_returned',arm='A',update=update,metrics=metrics,controller=controller.state_record());run.check()
            if update%16==0:
                current=decoder.frozen_fingerprint(prepared.model,prepared.lora_parameters)
                run.event('frozen_state_checked',update=update,initial_fingerprint=base,current_fingerprint=current)
                need(current==base,'frozen base/buffer changed')
        context.update(phase='final_state',update=None,generation_id=None)
        run.save_payload('A-final.pt',state_payload(prepared,controller));run.check()
        generation_index=0
        for row in document['dev']:
            for example in examples(row):
                case=cases[generation_index];generation_index+=1
                prompt=list(example.input_ids[:example.prompt_length])
                context.update(phase='generation',update=None,generation_id=generation_index)
                counts['generation_calls']+=1
                run.event('generation_begin',generation_id=generation_index,output_id=case['output_id'],source_id=case['source_id'],
                          profile_id=case['profile_id'],prompt_ids=prompt,max_new_tokens=80,use_cache=True)
                ids=list(prepared.greedy(prompt,max_new_tokens=80,use_cache=True));counts['generated_ids_returned']+=len(ids)
                eos=bool(ids and ids[-1]==prepared.model.config.eos_token_id)
                result=dict(case,token_ids=ids,text=decode(ids),scoring_text=decode(ids[:-1] if eos else ids),status='returned',natural_eos_observed=eos,
                            stop_reason='natural_eos' if eos else 'token_cap',token_cap_reached=len(ids)==80,
                            includes_eos_token=eos,generation_id=generation_index)
                run.save_json(case['output_id']+'.json',result)
                counts['generation_returns']+=1
                run.event('generation_returned',generation_id=generation_index,output_id=case['output_id'],token_ids=ids,
                          natural_eos_observed=eos,output_file=case['output_id']+'.json')
                run.check()
        context.update(phase='final_validation',generation_id=None)
        final_base=decoder.frozen_fingerprint(prepared.model,prepared.lora_parameters)
        run.event('frozen_state_checked',update=None,initial_fingerprint=base,current_fingerprint=final_base)
        need(final_base==base,'frozen base/buffer changed during inference')
        return run.finish('passed',final_check=final_check,counts=counts,semantic_outcomes_measured=False,
                          capacity_gate_passed=None,completed_output_ids=[c['output_id'] for c in cases])
    except BaseException as exc:
        try:
            if opened:run.event('update_failed',update=context['update'],controller=None if controller is None else controller.state_record(),exception_type=type(exc).__name__)
            run.event('capacity_exception',**context,exception_type=type(exc).__name__,message=str(exc),counts=counts)
            if prepared is not None and controller is not None:
                try:failure_snapshot=run.save_payload('A-failure.pt',state_payload(prepared,controller))
                except BaseException as failure:run.event('failure_snapshot_unavailable',exception_type=type(failure).__name__)
            return run.finish('failed',exception={'type':type(exc).__name__,'message':str(exc)},counts=counts,
                              controller=None if controller is None else controller.state_record(),failure_snapshot=failure_snapshot,
                              semantic_outcomes_measured=False,capacity_gate_passed=None)
        except BaseException as terminal_error:
            return {'status':'incomplete','counts':counts,'terminal_unavailable':type(terminal_error).__name__}
    finally:
        if prepared is not None and original_forward is not None:prepared.model.forward=original_forward
        if controller is not None and original_step is not None:controller.generator_optimizer.step=original_step
