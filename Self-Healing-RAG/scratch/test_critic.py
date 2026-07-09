import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add app directory to Python path
app_path = str(Path(__file__).resolve().parent.parent / "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from critic.models import CriticEvaluation
from critic.parser import enforce_decision_rules
from critic.critic import CriticAgent

class TestCriticRulesAndParser(unittest.TestCase):
    def test_enforce_decision_rules_pass(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            relevant=True,
            complete=True,
            hallucination=False,
            prompt_injection=False,
            confidence=0.85,
            decision="PASS",
            reason="All good."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "PASS")

    def test_enforce_decision_rules_fail_ungrounded(self):
        eval_obj = CriticEvaluation(
            grounded=False,
            relevant=True,
            complete=True,
            hallucination=False,
            prompt_injection=False,
            confidence=0.90,
            decision="PASS",
            reason="Not grounded."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")

    def test_enforce_decision_rules_fail_low_confidence(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            relevant=True,
            complete=True,
            hallucination=False,
            prompt_injection=False,
            confidence=0.75,
            decision="PASS",
            reason="Uncertain."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")

    def test_enforce_decision_rules_fail_hallucination(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            relevant=True,
            complete=True,
            hallucination=True,
            prompt_injection=False,
            confidence=0.95,
            decision="PASS",
            reason="Hallucinated."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")

    def test_enforce_decision_rules_fail_prompt_injection(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            relevant=True,
            complete=True,
            hallucination=False,
            prompt_injection=True,
            confidence=0.95,
            decision="PASS",
            reason="Injection."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")

class TestCriticAgentMocked(unittest.TestCase):
    def test_empty_inputs(self):
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)
        
        # Test empty context
        res = agent.evaluate("question", "", "answer")
        self.assertEqual(res["decision"], "FAIL")
        self.assertIn("Empty context", res["reason"])

        # Test empty answer
        res = agent.evaluate("question", "context", "")
        self.assertEqual(res["decision"], "FAIL")
        self.assertIn("Empty answer", res["reason"])

    def test_llm_failure_fallback(self):
        from langchain_core.runnables import Runnable
        
        class MockStructuredLLM(Runnable):
            def invoke(self, *args, **kwargs):
                raise Exception("API Error")
                
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MockStructuredLLM()
        
        agent = CriticAgent(mock_llm)
        res = agent.evaluate("question", "context", "answer")
        
        # Should retry and then return fallback
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["reason"], "Critic parsing failed")

if __name__ == "__main__":
    unittest.main()
