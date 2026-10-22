import copy,tempfile,unittest
from pathlib import Path
import data
try:import probe
except ImportError:probe=None

class ProbeTests(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(probe,'exploratory probe not implemented')
 def rows(self):
  return [{'group_id':1,'register':r,'answer':2,'generated':'Value: 2.'} for r in probe.REGISTERS]
 def test_fourth_only_changes_requested_label(self):
  g=data.build()['test'][0];before=copy.deepcopy(g);original=data.examples(g)[1]
  e=probe.fourth(g)
  self.assertEqual(e['messages'][0],original['messages'][0])
  self.assertEqual(e['messages'][1]['content'],original['messages'][1]['content'].replace('Requested register: working.','Requested register: concise.'))
  self.assertEqual(g,before);self.assertEqual(e['answer'],original['answer']);self.assertNotIn('concise',data.REGISTERS)
 def test_prompt_plan_preserves_original_three(self):
  groups=data.build()['test'];plan=probe.prompt_plan(groups)
  self.assertEqual(plan['original'],[e for g in groups for e in data.examples(g)])
  self.assertEqual(len(plan['original']),48);self.assertEqual(len(plan['fourth']),16)
 def test_all_four_correct(self):
  m=probe.metrics(self.rows(),probe.REGISTERS)
  self.assertEqual(m['all_registers_correct_count'],1);self.assertEqual(m['answer_change_count'],0);self.assertEqual(m['format_coverage_count'],4)
 def test_invalid_outputs_not_counted_as_answer_agreement(self):
  rows=self.rows();rows[-1]['generated']='2'
  m=probe.metrics(rows,probe.REGISTERS)
  self.assertEqual(m['invalid_group_count'],1);self.assertIsNone(m['valid_group_answer_change_rate']);self.assertEqual(m['all_registers_correct_count'],0)
 def test_conflicting_answers_recorded(self):
  rows=self.rows();rows[-1]['generated']='Value: 3.'
  m=probe.metrics(rows,probe.REGISTERS);self.assertEqual(m['answer_change_count'],1);self.assertEqual(m['strict_correct_count'],3)
 def test_duplicate_or_missing_register_refused(self):
  for rows in [self.rows()[:-1],self.rows()+self.rows()[:1]]:
   with self.assertRaises(ValueError):probe.metrics(rows,probe.REGISTERS)
 def test_output_dir_reuse_refused_before_loading(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);sentinel=out/'keep';sentinel.write_text('old')
   with self.assertRaises(FileExistsError):probe.run(Path('missing'),out)
   self.assertEqual(sentinel.read_text(),'old')

 def test_partial_generation_retained_on_interrupt(self):
  def interrupted_rows():
   yield {'generated':'Value: 2.'}
   raise TimeoutError('simulated interruption')
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'raw.jsonl'
   self.assertTrue(hasattr(probe,'retain'),'incremental retention not implemented')
   with self.assertRaises(TimeoutError):probe.retain(interrupted_rows(),path)
   import json
   self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],[{'generated':'Value: 2.'}])
