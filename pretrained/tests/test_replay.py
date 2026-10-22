import tempfile,unittest
from pathlib import Path
import replay

class ReplaySafetyTests(unittest.TestCase):
    def test_existing_receipt_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';p.write_text('keep')
            with self.assertRaises(FileExistsError):replay.replay('missing','missing','missing',p)
            self.assertEqual(p.read_text(),'keep')
