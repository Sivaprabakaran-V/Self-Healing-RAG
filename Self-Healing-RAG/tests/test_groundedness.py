import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add app directory to Python path
app_path = str(Path(__file__).resolve().parent.parent / "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from critic.models import CriticEvaluation
from critic.critic import CriticAgent

class TestGroundedness(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_structured_llm = MagicMock()
        self.mock_llm.with_structured_output.return_value = self.mock_structured_llm
        
    def test_grounded_answer_passes(self):
        # Prepare evaluation output representing a grounded answer
        evaluation = CriticEvaluation(
            grounded=True,
            grounded_score=0.95,
            relevant=True,
            relevance_score=0.90,
            complete=True,
            completeness_score=0.90,
            hallucination=False,
            hallucination_score=0.05,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.90,
            decision="PASS",
            failure_reason="NONE",
            reason="The answer is fully supported by the retrieved context."
        )
        self.mock_structured_llm.invoke.return_value = evaluation
        
        agent = CriticAgent(self.mock_llm)
        res = agent.evaluate("What is Apple's founding year?", "Apple was founded in 1976.", "Apple was founded in 1976.")
        
        self.assertEqual(res["decision"], "PASS")
        self.assertEqual(res["failure_reason"], "NONE")
        self.assertTrue(res["grounded"])
        self.assertEqual(res["grounded_score"], 0.95)

    def test_ungrounded_answer_fails(self):
        # Prepare evaluation output representing an ungrounded answer
        evaluation = CriticEvaluation(
            grounded=False,
            grounded_score=0.20,
            relevant=True,
            relevance_score=0.90,
            complete=True,
            completeness_score=0.90,
            hallucination=False,
            hallucination_score=0.10,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.85,
            decision="FAIL",
            failure_reason="LOW_GROUNDEDNESS",
            reason="The answer mentions Google but the context is about Apple."
        )
        self.mock_structured_llm.invoke.return_value = evaluation
        
        agent = CriticAgent(self.mock_llm)
        res = agent.evaluate("What is Apple's founding year?", "Apple was founded in 1976.", "Google was founded in 1998.")
        
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "LOW_GROUNDEDNESS")
        self.assertFalse(res["grounded"])
        self.assertEqual(res["grounded_score"], 0.20)

if __name__ == "__main__":
    unittest.main()
