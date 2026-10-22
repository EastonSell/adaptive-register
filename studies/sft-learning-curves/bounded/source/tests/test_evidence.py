import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'vendor'))
from evidence import reserve,append,check_coverage,summarize,atomic
from supervise import monitor
import data

class EvidenceTests(unittest.TestCase):
    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError): reserve(d)
    def test_falsified_prediction_saved_before_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out=reserve(Path(d)/'run')
            row={'group_id':64,'register':'working','prediction':1,'answer':2,'generated':'Value: 1.','target':'Value: 2.'}
            append(out/'generations.jsonl',row)
            with self.assertRaises(ValueError): check_coverage([row],[(64,'working'),(65,'working')])
            self.assertEqual(json.loads((out/'generations.jsonl').read_text()),row)
            self.assertEqual(summarize([row])['strict_correct'],0)
    def test_duplicates_and_missing_rejected(self):
        row={'group_id':64,'register':'working'}
        with self.assertRaises(ValueError):check_coverage([row,row],[(64,'working')])
        with self.assertRaises(ValueError):check_coverage([],[(64,'working')])
    def test_all_invalid_not_factual_stability(self):
        row={'group_id':64,'register':'working','prediction':None,'answer':1,'generated':'bad','target':'Value: 1.'}
        result=summarize([row]); self.assertIsNone(result['valid_group_answer_change_rate'])
        self.assertEqual(result['invalid_groups'],1)
    def test_interrupt_before_and_after_case(self):
        for after in (False,True):
            with self.subTest(after=after),tempfile.TemporaryDirectory() as d:
                out=reserve(Path(d)/'run')
                script="import sys,time;sys.path.insert(0,"+repr(str(ROOT))+");from evidence import append;"
                if after: script+="append("+repr(str(out/'raw.jsonl'))+",{'case':1,'prediction':False});"
                script+='time.sleep(10)'
                code=monitor([sys.executable,'-c',script],out,{'wall_seconds':.3,'cpu_seconds':1200,'rss_bytes':3*1024**3,'sampling_seconds':.01})
                self.assertEqual(code,124); self.assertEqual(json.loads((out/'status.json').read_text())['status'],'incomplete')
                self.assertEqual((out/'raw.jsonl').exists(),after)
                if after:self.assertFalse(json.loads((out/'raw.jsonl').read_text())['prediction'])
                self.assertTrue((out/'RESULTS-MANIFEST.json').exists())
    def test_failure_exit_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            out=reserve(Path(d)/'run')
            code=monitor([sys.executable,'-c','raise ValueError("injected")'],out,{'wall_seconds':2,'cpu_seconds':1200,'rss_bytes':3*1024**3,'sampling_seconds':.01})
            self.assertNotEqual(code,0); self.assertIn('injected',(out/'console.log').read_text())
    def test_data_and_exposure(self):
        splits=data.build(); self.assertEqual([len(splits[k]) for k in ('train','validation','test')],[64,16,16])
        self.assertFalse({tuple(g['record']) for g in splits['train']}&{tuple(g['record']) for g in splits['validation']})
        self.assertEqual(len([e for g in splits['train'] for e in data.examples(g)]),192)
        cfg=json.loads((ROOT/'protocol.json').read_text()); self.assertEqual(cfg['checkpoints'],[0,24,96,384])
        self.assertEqual(cfg['max_updates'],384)
    def test_parser_unchanged_strict(self):
        for text in ('Value: 1.','Value: 1. The lookup returns the stored scalar.'):
            self.assertEqual(data.parse_answer(text),1)
        for text in ('1','Value: 1 and 2','Value: -1','Value: 1.5'):
            self.assertIsNone(data.parse_answer(text))
    def test_runner_no_test_evaluation(self):
        source=(ROOT/'runner.py').read_text()
        self.assertNotIn("splits['test']",source)
        self.assertIn("('train','validation')",source)
        self.assertIn("training.loss(model,None,batch,0.,0.)",source)
        self.assertNotIn('best_step',source)


class StateTests(unittest.TestCase):
    def test_evaluation_restores_rng_and_module_modes_on_error(self):
        import torch, random
        from runner import preserve_evaluation_state
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.1))
        model.train(); model[0].eval()
        before=torch.get_rng_state().clone(); py_before=random.getstate(); modes=[m.training for m in model.modules()]
        with self.assertRaises(RuntimeError):
            with preserve_evaluation_state(model):
                torch.rand(5); random.random(); model.eval(); raise RuntimeError('injected')
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(py_before,random.getstate()); self.assertEqual(modes,[m.training for m in model.modules()])
    def test_tensor_comparison_reports_identical_original(self):
        import shutil
        from runner import compare_original
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); checkpoint=out/'checkpoint'; checkpoint.mkdir()
            shutil.copyfile(ROOT/'vendor/original-sft24.safetensors',checkpoint/'adapter_model.safetensors')
            compare_original(checkpoint,out)
            result=json.loads((out/'step24-original-comparison.json').read_text())
            self.assertTrue(result['all_tensors_identical'])

class SupervisionTests(unittest.TestCase):
    cfg={'wall_seconds':2,'cpu_seconds':1200,'rss_bytes':3*1024**3,'sampling_seconds':.01}
    def test_persistent_live_sample_failure_fails_closed(self):
        from unittest.mock import patch
        import time
        for error in (FileNotFoundError('injected missing proc'),ValueError('injected malformed accounting')):
            with self.subTest(error=error),tempfile.TemporaryDirectory() as d:
                out=reserve(Path(d)/'run'); start=time.monotonic()
                with patch('supervise.sample',side_effect=error) as mocked:
                    code=monitor([sys.executable,'-c','import time;time.sleep(10)'],out,self.cfg)
                self.assertNotEqual(code,0); self.assertLess(time.monotonic()-start,2)
                self.assertEqual(mocked.call_count,1)
                status=json.loads((out/'status.json').read_text())
                self.assertEqual(status['reason'],'accounting_failure'); self.assertNotEqual(status['returncode'],0)
                self.assertTrue((out/'RESULTS-MANIFEST.json').exists())
    def test_missing_and_malformed_rss_rejected(self):
        from supervise import parse_sample
        stat=Path(f'/proc/{os.getpid()}/stat').read_text()
        for status in ('Name: x\n','VmRSS: nope kB','VmRSS: 100 MB','VmRSS: 0 kB','VmRSS: -1 kB','VmRSS: 1 kB\nVmRSS: 1 kB'):
            with self.subTest(status=status),self.assertRaises(ValueError):parse_sample(stat,status)
        self.assertEqual(parse_sample(stat,'VmRSS: 42 kB')[1],42*1024)
    def test_malformed_cpu_rejected(self):
        from supervise import parse_sample
        with self.assertRaises((ValueError,IndexError)): parse_sample('invalid stat','VmRSS: 42 kB')
    def test_zero_exit_without_completion_is_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            out=reserve(Path(d)/'run')
            code=monitor([sys.executable,'-c','pass'],out,self.cfg)
            self.assertNotEqual(code,0)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'incomplete')
    def test_wall_deadline_independent_of_sample(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            out=reserve(Path(d)/'run')
            with patch('supervise.sample',side_effect=AssertionError('must not sample after deadline')):
                code=monitor([sys.executable,'-c','import time;time.sleep(10)'],out,dict(self.cfg,wall_seconds=0))
            self.assertEqual(code,124)
            self.assertEqual(json.loads((out/'status.json').read_text())['reason'],'wall_budget')
    def test_verified_child_exit_during_sample_is_distinguished(self):
        from unittest.mock import patch
        import time
        def delayed_error(pid):
            time.sleep(.1); raise FileNotFoundError('exit race')
        with tempfile.TemporaryDirectory() as d:
            out=reserve(Path(d)/'run')
            with patch('supervise.sample',side_effect=delayed_error):
                code=monitor([sys.executable,'-c','pass'],out,self.cfg)
            self.assertNotEqual(code,0)
            self.assertEqual(json.loads((out/'resources.jsonl').read_text())['event'],'verified_child_exit_during_sample')

if __name__=='__main__':unittest.main()
