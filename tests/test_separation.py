import unittest,re
import numpy as np
import core
import separation as s
class SeparationTests(unittest.TestCase):
 def test_parameter_budget(self):
  self.assertEqual(sum(v.size for v in s.initialize(1,32).values()),804)
  self.assertEqual(sum(v.size for v in s.initialize(1,46).values()),1154)
 def test_gradient(self):
  d=core.dataset(1)[0];x=d['x'][:9,:20];a=d['answer'][:9]
  p=s.initialize(1,5);loss,g=s.objective(p,x,a)
  for k,v in p.items():
   idx=tuple(n-1 for n in v.shape);old=v[idx]
   v[idx]=old+1e-5;u=s.objective(p,x,a)[0]
   v[idx]=old-1e-5;l=s.objective(p,x,a)[0];v[idx]=old
   self.assertAlmostEqual(g[k][idx],(u-l)/2e-5,places=6)
 def test_register_cannot_change_content(self):
  d=core.dataset(2)[2];p=s.initialize(3)
  a=s.predict(p,d['x']);changed=d['x'].copy();changed[:,-3:]=np.random.default_rng(2).normal(size=changed[:,-3:].shape)*100
  np.testing.assert_array_equal(a,s.predict(p,changed))
  self.assertEqual(s.evaluate(p,d)['answer_change_rate'],0.)
 def test_formatting_does_not_correct_wrong_prediction(self):
  for prediction in range(4):
   for r in core.REGISTERS:
    text=s.format_prediction(prediction,r)
    self.assertEqual(re.findall(r'\d+',text),[str(prediction)])
 def test_repeat_learning(self):
  a,b,_=core.dataset(3)
  p,h=s.fit(a,b,seed=2,epochs=6);q,k=s.fit(a,b,seed=2,epochs=6)
  for name in p:np.testing.assert_array_equal(p[name],q[name])
  self.assertEqual(h,k);self.assertLess(h[-1],h[0])
 def test_validation_selection(self):
  a,b,_=core.dataset(3)
  p,h=s.fit(a,b,seed=2,epochs=6)
  self.assertAlmostEqual(s.objective(p,b['x'][:,:20],b['answer'])[0],min(h))

class RunnerTests(unittest.TestCase):
 def test_grid_and_paired_results(self):
  import run_separation as runner
  result=runner.experiment(dataset_seeds=(8,),model_seeds=(2,),epochs=2)
  self.assertEqual(len(result['runs']),6)
  self.assertEqual(len(result['paired_accuracy_differences']),2)
  self.assertTrue(all(r['active_parameters'] in [804,1068,1154,1164,1263] for r in result['runs']))

 def test_dead_register_weights_excluded(self):
  import run_separation as runner
  rows=runner.experiment(dataset_seeds=(8,),model_seeds=(2,),epochs=1)["runs"]
  u=next(r for r in rows if r["objective"]=="unconditioned")
  self.assertEqual(u["structural_nonaux_parameters"],1164)
  self.assertEqual(u["active_parameters"],1068)
