"""
Critic Parser (V3-stabilised)

Issue 4 fix: adds extract_json_from_text() for the JSON fallback path
used when structured function-calling fails.

Issue 3 fix: get_fallback_result() no longer generates evaluation_id or
timestamp — CriticAgent.evaluate() injects those after parsing.
"""
import json
import logging
import re
from typing import Any, Dict, Optional

from critic.models import CriticEvaluation
from critic.exceptions import ParserError

logger = logging.getLogger("SelfHealingRAG.Critic.Parser")


class StructuredParser:
    """Parses and validates CriticEvaluation models from LLM outputs or dicts."""

    @staticmethod
    def parse_to_model(data: Any) -> CriticEvaluation:
        """
        Converts the incoming structured output or dictionary into a
        CriticEvaluation model.

        Raises:
            ParserError: If validation fails for any reason.
        """
        try:
            if isinstance(data, CriticEvaluation):
                return data
            if isinstance(data, dict):
                # Strip keys that no longer exist in the model (e.g., stale
                # evaluation_id / timestamp from old cached outputs)
                allowed = set(CriticEvaluation.model_fields.keys())
                cleaned = {k: v for k, v in data.items() if k in allowed}
                return CriticEvaluation(**cleaned)
            raise ValueError(
                "Input data must be a dictionary or a CriticEvaluation instance."
            )
        except Exception as exc:
            raise ParserError(
                f"Failed to parse and validate output as CriticEvaluation: {exc}"
            ) from exc

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts the first JSON object found in a raw LLM text response.

        Used as the fallback when structured function-calling fails (Issue 4).
        Scans for the outermost `{...}` block and attempts to parse it.

        Args:
            text: Raw LLM text response (may contain prose before/after JSON).

        Returns:
            Parsed dict if a valid JSON object is found, otherwise None.
        """
        if not text:
            return None

        # Find outermost JSON object (greedy, dotall)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.debug("extract_json_from_text: no JSON object found in text.")
            return None

        candidate = match.group(0)
        
        # Try normal JSON parsing first
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                logger.debug(
                    "extract_json_from_text: successfully parsed JSON "
                    "(%d keys).", len(parsed)
                )
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to clean common malformations
        try:
            # 1. Remove comments (lines starting with // or enclosed in /* ... */)
            cleaned = re.sub(r"/\*.*?\*/", "", candidate, flags=re.DOTALL)
            cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)
            
            # 2. Remove trailing commas before closing braces/brackets
            cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
            
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                logger.debug(
                    "extract_json_from_text: successfully parsed cleaned JSON "
                    "(%d keys).", len(parsed)
                )
                return parsed
        except Exception as exc:
            logger.debug(
                "extract_json_from_text: Cleaned JSON parse failed — %s", exc
            )
            
        return None

    @staticmethod
    def get_fallback_result(
        failure_reason: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Constructs a safe fallback dictionary conforming to the structured
        output layout.

        Note: evaluation_id, timestamp, and execution_time_s are NOT included
        here. They are injected by CriticAgent.evaluate() after this method
        returns (Issue 3).
        """
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
            "reason": reason,
        }


def enforce_decision_rules(evaluation: CriticEvaluation) -> CriticEvaluation:
    """
    Backwards-compatible wrapper function for business rules validation.
    Kept for any external callers; does not pass answer text.
    """
    logger.info("Calling backwards-compatible enforce_decision_rules wrapper...")
    from critic.rules import BusinessRuleValidator
    validator = BusinessRuleValidator()
    return validator.validate(evaluation, answer="")
