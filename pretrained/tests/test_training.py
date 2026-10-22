import unittest,os
import torch
from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM
from peft import LoraConfig,get_peft_model
import data,training

class TrainingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.t=AutoTokenizer.from_pretrained(os.environ['MODEL_DIR'],local_files_only=True,trust_remote_code=False)
 def test_mask_answer_alignment(self):
  e=data.examples(data.build()['train'][0]);b=training.encode(self.t,e)
  for i in range(3):
   a=b['answer_positions'][i];self.assertEqual(b['labels'][i,a].item(),b['digit_ids'][e[i]['answer']])
   self.assertTrue(torch.all(b['labels'][i,:b['prompt_lengths'][i]]==-100))
   self.assertTrue(torch.all(b['labels'][i,b['attention_mask'][i]==0]==-100))
   self.assertTrue(b['aux_mask'][i].sum()>0)
 def test_backprop_changes_adapters_not_base(self):
  torch.manual_seed(11);torch.set_num_threads(2)
  cfg=LlamaConfig(vocab_size=len(self.t),hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
  m=get_peft_model(LlamaForCausalLM(cfg),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',lora_dropout=0.))
  head=torch.nn.Linear(16,3);b=training.encode(self.t,data.examples(data.build()['train'][0]))
  frozen=[p for p in m.parameters() if not p.requires_grad]
  before=[p.detach().clone() for p in frozen]
  loss,parts=training.loss(m,head,b,.1,1.);self.assertTrue(torch.isfinite(loss))
  loss.backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters() if p.requires_grad))
  torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]+list(head.parameters()),lr=.001).step()
  for a,p in zip(before,frozen):self.assertTrue(torch.equal(a,p))
 def test_consistency_zero_for_identical_distributions(self):
  z=torch.randn(1,4).repeat(3,1);self.assertEqual(training.consistency(z).item(),0.)

class EvaluationTests(unittest.TestCase):
 def test_invalid_and_conflicting_generations_count_wrong(self):
  import pilot
  rows=[{'group_id':1,'answer':2,'prediction':2,'exact_template_match':True},
        {'group_id':1,'answer':2,'prediction':None,'exact_template_match':False},
        {'group_id':1,'answer':2,'prediction':3,'exact_template_match':False}]
  s=pilot.summarize(rows)
  self.assertAlmostEqual(s['answer_accuracy'],1/3)
  self.assertEqual(s['groups_with_invalid_generation'],1)
  self.assertEqual(s['all_registers_correct'],0.)

class OutputSafetyTests(unittest.TestCase):
 def test_existing_output_refused_without_touching_files(self):
  import tempfile
  from pathlib import Path
  import pilot
  with tempfile.TemporaryDirectory() as d:
   output=Path(d)/'run';output.mkdir();sentinel=output/'old.json';sentinel.write_text('keep')
   with self.assertRaises(FileExistsError):pilot.run(Path(d)/'absent_model',output,('base',))
   self.assertEqual(sentinel.read_text(),'keep')
