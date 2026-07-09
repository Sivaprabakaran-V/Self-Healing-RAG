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
      - prompt_injection == False
      - confidence >= 0.80
    Otherwise:
      - decision = FAIL
    
    This function modifies the decision field in-place if it violates the rules
    to prevent LLM inconsistencies.
    """
    expected_decision = "PASS"
    
    if not evaluation.grounded:
        expected_decision = "FAIL"
        logger.info("Decision rule triggered: 'grounded' is False. Setting decision to FAIL.")
    elif evaluation.hallucination:
        expected_decision = "FAIL"
        logger.info("Decision rule triggered: 'hallucination' is True. Setting decision to FAIL.")
    elif evaluation.prompt_injection:
        expected_decision = "FAIL"
        logger.info("Decision rule triggered: 'prompt_injection' is True. Setting decision to FAIL.")
    elif evaluation.confidence < 0.80:
        expected_decision = "FAIL"
        logger.info(f"Decision rule triggered: 'confidence' is {evaluation.confidence} (less than 0.80). Setting decision to FAIL.")
        
    if evaluation.decision != expected_decision:
        logger.warning(
            f"Critic LLM returned decision '{evaluation.decision}', but rules required "
            f"'{expected_decision}'. Overriding decision to '{expected_decision}'."
        )
        evaluation.decision = expected_decision
        
    return evaluation
