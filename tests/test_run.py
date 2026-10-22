import json, tempfile, unittest
from pathlib import Path
import run
class RunTests(unittest.TestCase):
 def test_artifacts_and_all_ablations(self):
  with tempfile.TemporaryDirectory() as d:
   report=run.experiment(Path(d),seeds=(1,),epochs=2)
   self.assertEqual(len(report['runs']),4)
   self.assertEqual(report['status'],'new_recovery_rebuild_not_original_reproduction')
   self.assertEqual(len(list(Path(d).glob('*.npz'))),4)
   self.assertTrue((Path(d)/'summary.json').is_file())
   self.assertEqual(json.loads((Path(d)/'summary.json').read_text()),report)
