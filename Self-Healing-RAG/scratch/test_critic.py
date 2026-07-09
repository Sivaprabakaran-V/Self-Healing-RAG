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
            grounded_score=0.9,
            relevant=True,
            relevance_score=0.9,
            complete=True,
            completeness_score=0.9,
            hallucination=False,
            hallucination_score=0.1,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.85,
            decision="PASS",
            failure_reason="NONE",
            reason="All good."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "PASS")
        self.assertEqual(enforced.failure_reason, "NONE")

    def test_enforce_decision_rules_fail_ungrounded(self):
        eval_obj = CriticEvaluation(
            grounded=False,
            grounded_score=0.3,
            relevant=True,
            relevance_score=0.9,
            complete=True,
            completeness_score=0.9,
            hallucination=False,
            hallucination_score=0.1,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.90,
            decision="PASS",
            failure_reason="NONE",
            reason="Not grounded."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")
        self.assertEqual(enforced.failure_reason, "LOW_GROUNDEDNESS")

    def test_enforce_decision_rules_fail_hallucination(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            grounded_score=0.9,
            relevant=True,
            relevance_score=0.9,
            complete=True,
            completeness_score=0.9,
            hallucination=True,
            hallucination_score=0.8,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.95,
            decision="PASS",
            failure_reason="NONE",
            reason="Hallucinated."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")
        self.assertEqual(enforced.failure_reason, "HALLUCINATION")

    def test_enforce_decision_rules_fail_prompt_injection(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            grounded_score=0.9,
            relevant=True,
            relevance_score=0.9,
            complete=True,
            completeness_score=0.9,
            hallucination=False,
            hallucination_score=0.1,
            prompt_injection_attempt=True,
            prompt_injection_success=True,
            overall_confidence=0.95,
            decision="PASS",
            failure_reason="NONE",
            reason="Injection succeeded."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")
        self.assertEqual(enforced.failure_reason, "PROMPT_INJECTION")

    def test_enforce_decision_rules_fail_low_relevance(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            grounded_score=0.9,
            relevant=False,
            relevance_score=0.3,
            complete=True,
            completeness_score=0.9,
            hallucination=False,
            hallucination_score=0.1,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.95,
            decision="PASS",
            failure_reason="NONE",
            reason="Irrelevant."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")
        self.assertEqual(enforced.failure_reason, "LOW_RELEVANCE")

    def test_enforce_decision_rules_fail_incomplete(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            grounded_score=0.9,
            relevant=True,
            relevance_score=0.9,
            complete=False,
            completeness_score=0.3,
            hallucination=False,
            hallucination_score=0.1,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.95,
            decision="PASS",
            failure_reason="NONE",
            reason="Incomplete."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")
        self.assertEqual(enforced.failure_reason, "INCOMPLETE_ANSWER")

    def test_enforce_decision_rules_fail_low_confidence(self):
        eval_obj = CriticEvaluation(
            grounded=True,
            grounded_score=0.9,
            relevant=True,
            relevance_score=0.9,
            complete=True,
            completeness_score=0.9,
            hallucination=False,
            hallucination_score=0.1,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.75,
            decision="PASS",
            failure_reason="NONE",
            reason="Uncertain."
        )
        enforced = enforce_decision_rules(eval_obj)
        self.assertEqual(enforced.decision, "FAIL")
        self.assertEqual(enforced.failure_reason, "LOW_CONFIDENCE")

class TestCriticAgentMocked(unittest.TestCase):
    def test_empty_inputs(self):
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)
        
        # Test empty context
        res = agent.evaluate("question", "", "answer")
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "EMPTY_CONTEXT")
        self.assertIn("Empty context", res["reason"])

        # Test empty answer
        res = agent.evaluate("question", "context", "")
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "INCOMPLETE_ANSWER")
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
        self.assertEqual(res["failure_reason"], "PARSER_ERROR")
        self.assertIn("Critic parsing failed", res["reason"])

if __name__ == "__main__":
    unittest.main()
