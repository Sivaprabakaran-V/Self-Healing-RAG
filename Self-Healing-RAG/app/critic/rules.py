from critic.models import CriticEvaluation

class BusinessRuleValidator:
    """Enforces deterministic business rules to validate LLM evaluations."""
    def __init__(
        self,
        confidence_threshold: float = 0.80,
        groundedness_threshold: float = 0.80,
        relevance_threshold: float = 0.80,
        completeness_threshold: float = 0.80
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.groundedness_threshold = groundedness_threshold
        self.relevance_threshold = relevance_threshold
        self.completeness_threshold = completeness_threshold

    def validate(self, evaluation: CriticEvaluation) -> CriticEvaluation:
        """
        Applies deterministic rules to the evaluation.
        Overrides decision and failure_reason in-place if rules fail.
        """
        expected_decision = "PASS"
        expected_failure_reason = "NONE"

        if evaluation.prompt_injection_success:
            expected_decision = "FAIL"
            expected_failure_reason = "PROMPT_INJECTION"
        elif evaluation.hallucination:
            expected_decision = "FAIL"
            expected_failure_reason = "HALLUCINATION"
        elif not evaluation.grounded or evaluation.grounded_score < self.groundedness_threshold:
            expected_decision = "FAIL"
            expected_failure_reason = "LOW_GROUNDEDNESS"
        elif not evaluation.relevant or evaluation.relevance_score < self.relevance_threshold:
            expected_decision = "FAIL"
            expected_failure_reason = "LOW_RELEVANCE"
        elif not evaluation.complete or evaluation.completeness_score < self.completeness_threshold:
            expected_decision = "FAIL"
            expected_failure_reason = "INCOMPLETE_ANSWER"
        elif evaluation.overall_confidence < self.confidence_threshold:
            expected_decision = "FAIL"
            expected_failure_reason = "LOW_CONFIDENCE"

        # Apply override if needed
        if evaluation.decision != expected_decision or evaluation.failure_reason != expected_failure_reason:
            evaluation.decision = expected_decision
            evaluation.failure_reason = expected_failure_reason

        return evaluation
