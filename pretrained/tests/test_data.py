import unittest
import data
class DataTests(unittest.TestCase):
 def test_disjoint_and_grouped(self):
  s=data.build();sets=[{tuple(r['record']) for r in part} for part in s.values()]
  self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
  self.assertEqual([len(x) for x in s.values()],[64,16,16])
  self.assertEqual(data.build(),s)
  for group in s['test']:
   examples=data.examples(group)
   self.assertEqual(len(examples),3)
   self.assertEqual(len({e['answer'] for e in examples}),1)
 def test_parser_rejects_conflicts(self):
  self.assertEqual(data.parse_answer('Value: 2. Stored scalar.'),2)
  for text in ['2','Value: 2 but 3','Value: 20','Value: -2','Value: 1.5','Value: 2. Value: 2.']:self.assertIsNone(data.parse_answer(text))
 def test_target_roundtrip(self):
  for record in data.build()['train']:
   for e in data.examples(record):self.assertEqual(data.parse_answer(e['target']),e['answer'])
