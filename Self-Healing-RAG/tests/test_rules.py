import sys
import unittest
from pathlib import Path

# Add app directory to Python path
app_path = str(Path(__file__).resolve().parent.parent / "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from critic.models import CriticEvaluation
from critic.rules import BusinessRuleValidator

class TestRules(unittest.TestCase):
    def test_override_llm_decision_from_pass_to_fail(self):
        # LLM says PASS, but hallucination is True -> should override to FAIL / HALLUCINATION
        evaluation = CriticEvaluation(
            grounded=True,
            grounded_score=0.90,
            relevant=True,
            relevance_score=0.90,
            complete=True,
            completeness_score=0.90,
            hallucination=True,
            hallucination_score=0.85,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.90,
            decision="PASS",
            failure_reason="NONE",
            reason="Llm incorrectly decided to pass."
        )
        
        validator = BusinessRuleValidator()
        validated = validator.validate(evaluation)
        
        self.assertEqual(validated.decision, "FAIL")
        self.assertEqual(validated.failure_reason, "HALLUCINATION")

    def test_configurable_thresholds_override(self):
        # With default confidence_threshold=0.80, a score of 0.85 passes.
        # But if we configure confidence_threshold=0.90, a score of 0.85 should fail!
        evaluation = CriticEvaluation(
            grounded=True,
            grounded_score=0.90,
            relevant=True,
            relevance_score=0.95,
            complete=True,
            completeness_score=0.95,
            hallucination=False,
            hallucination_score=0.05,
            prompt_injection_attempt=False,
            prompt_injection_success=False,
            overall_confidence=0.85,
            decision="PASS",
            failure_reason="NONE",
            reason="Test confidence threshold override"
        )
        
        # Test default passes
        validator_default = BusinessRuleValidator()
        self.assertEqual(validator_default.validate(evaluation).decision, "PASS")
        
        # Test strict configuration fails
        validator_strict = BusinessRuleValidator(confidence_threshold=0.90)
        validated_strict = validator_strict.validate(evaluation)
        self.assertEqual(validated_strict.decision, "FAIL")
        self.assertEqual(validated_strict.failure_reason, "LOW_CONFIDENCE")

    def test_pydantic_model_consistency_rules(self):
        # Pydantic validation should fail if decision is PASS but failure_reason is not NONE
        with self.assertRaises(ValueError):
            CriticEvaluation(
                grounded=True,
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
                failure_reason="LOW_CONFIDENCE",  # Inconsistent
                reason="Invalid consistency"
            )

        # Pydantic validation should fail if decision is FAIL but failure_reason is NONE
        with self.assertRaises(ValueError):
            CriticEvaluation(
                grounded=True,
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
                decision="FAIL",
                failure_reason="NONE",  # Inconsistent
                reason="Invalid consistency"
            )

if __name__ == "__main__":
    unittest.main()
