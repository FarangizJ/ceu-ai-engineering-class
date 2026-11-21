import unittest
from scoring import check_answer


class AnswerChecks(unittest.TestCase):
    def test_correct_fact_needs_correct_page(self):
        case = {"fact": "30 ml", "page": "[catalog.pdf p.1]"}
        self.assertTrue(check_answer(case, "30 ml [catalog.pdf p.1]", True))
        self.assertFalse(check_answer(case, "30 ml [catalog.pdf p.2]", True))
        self.assertFalse(check_answer(case, "130 ml [catalog.pdf p.1]", True))

    def test_missing_evidence_requires_abstention(self):
        case = {"fact": "LUMI-81", "page": "[insert.pdf p.1]"}
        self.assertTrue(check_answer(case, "Insufficient evidence.", False))
        self.assertFalse(check_answer(case, "LUMI-81 [insert.pdf p.1]", False))

    def test_clarification(self):
        case = {"expect": "clarify which product"}
        self.assertTrue(check_answer(case, "Which product do you mean?"))
        self.assertFalse(check_answer(case, "30 ml."))
