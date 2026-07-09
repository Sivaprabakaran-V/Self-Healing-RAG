import json
import logging
from typing import Any, Dict
from critic.models import CriticEvaluation
from critic.exceptions import ParserError

logger = logging.getLogger("SelfHealingRAG.Critic.Parser")

class StructuredParser:
    """Parses and validates CriticEvaluation models from LLM outputs or dicts."""
    
    @staticmethod
    def parse_to_model(data: Any) -> CriticEvaluation:
        """
        Converts the incoming structured output or dictionary into a CriticEvaluation model.
        Raises ParserError if validation fails.
        """
        try:
            if isinstance(data, CriticEvaluation):
                return data
            if isinstance(data, dict):
                return CriticEvaluation(**data)
            raise ValueError("Input data must be a dictionary or a CriticEvaluation instance.")
        except Exception as e:
            raise ParserError(f"Failed to parse and validate output as CriticEvaluation: {e}") from e

    @staticmethod
    def get_fallback_result(
        failure_reason: str,
        reason: str
    ) -> Dict[str, Any]:
        """Constructs a raw fallback dictionary conforming to the structured output layout."""
        return {
            "grounded": False,
            "grounded_score": 0.0,
            "relevant": False,
            "relevance_score": 0.0,
            "complete": False,
            "completeness_score": 0.0,
            "hallucination": False,
            "hallucination_score": 0.0,
            "prompt_injection_attempt": False,
            "prompt_injection_success": False,
            "overall_confidence": 0.0,
            "decision": "FAIL",
            "failure_reason": failure_reason,
            "reason": reason
        }

def enforce_decision_rules(evaluation: CriticEvaluation) -> CriticEvaluation:
    """
    Backwards-compatible wrapper function for business rules validation.
    """
    logger.info("Calling backwards-compatible enforce_decision_rules wrapper...")
    from critic.rules import BusinessRuleValidator
    validator = BusinessRuleValidator()
    return validator.validate(evaluation)
