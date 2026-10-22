"""Checks the teaching templates, not model comprehension or factual accuracy."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name("content_presentation.py")


class ContentPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = None
        if PATH.exists():
            spec = importlib.util.spec_from_file_location("content_presentation", PATH)
            cls.demo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.demo)

    def setUp(self):
        self.assertIsNotNone(self.demo, "Teaching module is not implemented yet")

    def test_each_template_retains_all_three_literals(self):
        for register in self.demo.REGISTERS:
            text = self.demo.render(self.demo.FACTS, register)
            self.assertEqual(self.demo.missing_literals(self.demo.FACTS, text), [])

    def test_shorter_example_omits_required_literals(self):
        text = "Check in this evening. Bring your ID."
        self.assertEqual(self.demo.missing_literals(self.demo.FACTS, text),
                         ["Bay 3", "18:45", "printed ID"])

    def test_consistently_wrong_content_stays_wrong(self):
        wrong = dict(self.demo.FACTS, place="Bay 9")
        for register in self.demo.REGISTERS:
            text = self.demo.render(wrong, register)
            self.assertIn("Bay 9", text)
            self.assertEqual(self.demo.missing_literals(self.demo.FACTS, text), ["Bay 3"])

    def test_default_is_working(self):
        self.assertEqual(self.demo.render(self.demo.FACTS),
                         self.demo.render(self.demo.FACTS, "working"))

    def test_unknown_register_is_rejected(self):
        with self.assertRaises(ValueError):
            self.demo.render(self.demo.FACTS, "clever")

    def test_incomplete_record_is_rejected(self):
        with self.assertRaises(ValueError):
            self.demo.render({"place": "Bay 3"})

    def test_blank_fact_is_rejected(self):
        with self.assertRaises(ValueError):
            self.demo.render(dict(self.demo.FACTS, time=" "))


if __name__ == "__main__":
    unittest.main()
