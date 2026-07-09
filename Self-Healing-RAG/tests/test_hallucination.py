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

class TestHallucination(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_structured_llm = MagicMock()
        self.mock_llm.with_structured_output.return_value = self.mock_structured_llm

    def test_safe_refusal_passes_without_hallucination_flag(self):
        # A refusal like "I do not have enough information" is not a hallucination
        evaluation = CriticEvaluation(
            grounded=True,  # Refusal is grounded in lack of information
            grounded_score=0.90,
            relevant=True,
            relevance_score=0.90,
            complete=True,
            completeness_score=0.90,
            hallucination=False,
            hallucination_score=0.0,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.90,
            decision="PASS",
            failure_reason="NONE",
            reason="The answer is a safe refusal because the retrieved context does not contain Microsoft's founding year."
        )
        self.mock_structured_llm.invoke.return_value = evaluation
        
        agent = CriticAgent(self.mock_llm)
        res = agent.evaluate("When was Microsoft founded?", "Apple was founded in 1976.", "I could not find sufficient information in the retrieved context.")
        
        self.assertEqual(res["decision"], "PASS")
        self.assertEqual(res["failure_reason"], "NONE")
        self.assertFalse(res["hallucination"])
        self.assertEqual(res["hallucination_score"], 0.0)

    def test_hallucinated_answer_fails(self):
        # Introduced unsupported facts about Microsoft
        evaluation = CriticEvaluation(
            grounded=False,
            grounded_score=0.10,
            relevant=True,
            relevance_score=0.90,
            complete=True,
            completeness_score=0.90,
            hallucination=True,
            hallucination_score=0.95,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.85,
            decision="FAIL",
            failure_reason="HALLUCINATION",
            reason="The answer introduces unsupported factual claims about Microsoft when context only contains Apple information."
        )
        self.mock_structured_llm.invoke.return_value = evaluation
        
        agent = CriticAgent(self.mock_llm)
        res = agent.evaluate("When was Microsoft founded?", "Apple was founded in 1976.", "Microsoft was founded in 1975.")
        
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "HALLUCINATION")
        self.assertTrue(res["hallucination"])
        self.assertEqual(res["hallucination_score"], 0.95)

if __name__ == "__main__":
    unittest.main()
