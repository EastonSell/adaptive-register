import unittest, tempfile
from pathlib import Path
import numpy as np
import core

class CoreTests(unittest.TestCase):
 def test_split_disjoint_and_counterfactuals(self):
  splits=core.dataset(7)
  keys=[set(map(tuple,s['keys'][:,:4])) for s in splits]
  self.assertFalse(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
  for s in splits:
   self.assertEqual(s['x'].shape[1],23)
   np.testing.assert_array_equal(s['answer'][::3],s['answer'][1::3])
   np.testing.assert_array_equal(s['answer'][::3],s['answer'][2::3])
   np.testing.assert_array_equal(s['register'].reshape(-1,3),np.tile(np.arange(3),(len(s['keys']),1)))
 def test_seed_repeats(self):
  for a,b in zip(core.dataset(3),core.dataset(3)):np.testing.assert_array_equal(a['x'],b['x'])
 def test_gradient(self):
  s=core.dataset(1)[0]; x=s['x'][:12];a=s['answer'][:12];r=s['register'][:12]
  p=core.initialize(2,5);loss,g=core.objective(p,x,a,r,.3,2)
  self.assertTrue(np.isfinite(loss))
  for name,v in p.items():
   for idx in [tuple(0 for _ in v.shape),tuple(n-1 for n in v.shape)]:
    old=v[idx];v[idx]=old+1e-5;plus=core.objective(p,x,a,r,.3,2)[0]
    v[idx]=old-1e-5;minus=core.objective(p,x,a,r,.3,2)[0];v[idx]=old
    self.assertAlmostEqual(g[name][idx],(plus-minus)/2e-5,places=6)
 def test_learning_and_reproducibility(self):
  train,val,_=core.dataset(2)
  a,h=core.fit(train,val,seed=3,epochs=12)
  b,k=core.fit(train,val,seed=3,epochs=12)
  self.assertLess(h[-1]['train_loss'],h[0]['train_loss'])
  for name in a:np.testing.assert_array_equal(a[name],b[name])
  self.assertEqual(h,k)
 def test_checkpoint(self):
  p=core.initialize(5)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'m.npz';core.save(p,f);q=core.load(f)
   for name in p:np.testing.assert_array_equal(p[name],q[name])
 def test_unconditioned_invariance(self):
  s=core.dataset(1)[2];x=s['x'].copy();x[:,-3:]=0
  a,r=core.predict(core.initialize(3),x)
  np.testing.assert_array_equal(a[::3],a[1::3]);np.testing.assert_array_equal(a[::3],a[2::3])
 def test_invalid_register(self):
  with self.assertRaises(ValueError):core.render(2,'expert')
 def test_renderer(self):
  for r in core.REGISTERS:self.assertIn('2',core.render(2,r))
if __name__=='__main__':unittest.main()
