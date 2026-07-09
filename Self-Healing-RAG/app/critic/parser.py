import logging
from critic.models import CriticEvaluation

logger = logging.getLogger("SelfHealingRAG.Critic.Parser")

def enforce_decision_rules(evaluation: CriticEvaluation) -> CriticEvaluation:
    """
    Enforces the strict PASS/FAIL decision rules on a CriticEvaluation instance.
    
    Decision Rules:
    PASS if:
      - grounded == True
      - hallucination == False
      - prompt_injection_success == False
      - relevant == True
      - complete == True
      - overall_confidence >= 0.80
    Otherwise:
      - decision = FAIL
    
    This function modifies the decision and failure_reason fields in-place if it violates the rules
    to prevent LLM inconsistencies.
    """
    expected_decision = "PASS"
    expected_failure_reason = "NONE"
    
    if evaluation.prompt_injection_success:
        expected_decision = "FAIL"
        expected_failure_reason = "PROMPT_INJECTION"
        logger.info("Decision rule triggered: 'prompt_injection_success' is True. Setting decision to FAIL and failure_reason to PROMPT_INJECTION.")
    elif evaluation.hallucination:
        expected_decision = "FAIL"
        expected_failure_reason = "HALLUCINATION"
        logger.info("Decision rule triggered: 'hallucination' is True. Setting decision to FAIL and failure_reason to HALLUCINATION.")
    elif not evaluation.grounded:
        expected_decision = "FAIL"
        expected_failure_reason = "LOW_GROUNDEDNESS"
        logger.info("Decision rule triggered: 'grounded' is False. Setting decision to FAIL and failure_reason to LOW_GROUNDEDNESS.")
    elif not evaluation.relevant:
        expected_decision = "FAIL"
        expected_failure_reason = "LOW_RELEVANCE"
        logger.info("Decision rule triggered: 'relevant' is False. Setting decision to FAIL and failure_reason to LOW_RELEVANCE.")
    elif not evaluation.complete:
        expected_decision = "FAIL"
        expected_failure_reason = "INCOMPLETE_ANSWER"
        logger.info("Decision rule triggered: 'complete' is False. Setting decision to FAIL and failure_reason to INCOMPLETE_ANSWER.")
    elif evaluation.overall_confidence < 0.80:
        expected_decision = "FAIL"
        expected_failure_reason = "LOW_CONFIDENCE"
        logger.info(f"Decision rule triggered: 'overall_confidence' is {evaluation.overall_confidence} (less than 0.80). Setting decision to FAIL and failure_reason to LOW_CONFIDENCE.")
        
    if evaluation.decision != expected_decision or evaluation.failure_reason != expected_failure_reason:
        logger.warning(
            f"Critic LLM returned decision '{evaluation.decision}' with failure_reason '{evaluation.failure_reason}', "
            f"but rules required decision '{expected_decision}' with failure_reason '{expected_failure_reason}'. "
            f"Overriding decision and failure_reason."
        )
        evaluation.decision = expected_decision
        evaluation.failure_reason = expected_failure_reason
        
    return evaluation
