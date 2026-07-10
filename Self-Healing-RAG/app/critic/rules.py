"""
Business Rule Validator (V3-stabilised)

Enforces deterministic rules that override LLM evaluation output.

Issue 1 fix: adds a lexical prompt injection success detector.
When prompt_injection_attempt=True, the generated answer is scanned
for instruction-following signals. If any are found, prompt_injection_success
is forced to True and the decision is overridden to FAIL regardless of
what the LLM returned.
"""
import re
import logging
from typing import Optional

from critic.models import CriticEvaluation

logger = logging.getLogger("SelfHealingRAG.Critic.Rules")

# ---------------------------------------------------------------------------
# Deterministic prompt injection success signals
#
# These are lower-cased patterns that indicate the generator followed or
# partially followed a malicious instruction.  The list covers:
#   - persona / role adoption
#   - system prompt / instruction disclosure
#   - explicit acknowledgement of the attacker's command
#   - style or behaviour override markers
# ---------------------------------------------------------------------------
_INJECTION_SUCCESS_SIGNALS: tuple[str, ...] = (
    # Persona / role adoption
    "i am now",
    "i'm now",
    "as you requested, i will",
    "as instructed",
    "new persona",
    "you are now",
    "act as",
    "pretend",
    "roleplay",
    "i will now act",
    "switching to",
    "entering",
    # System / instruction disclosure
    "my system prompt",
    "my instructions are",
    "here are my instructions",
    "my prompt is",
    "my original prompt",
    "my rules are",
    "my constraints are",
    "the system prompt says",
    "my internal prompt",
    "i was instructed to",
    "i was told to",
    "my directive is",
    "reveal my prompt",
    # Explicit acknowledgement of override command
    "ignoring previous instructions",
    "forget previous instructions",
    "disregarding previous",
    "overriding my",
    "bypassing",
    "jailbreak",
    "as the attacker requested",
    "as you said, i will ignore",
    # Compliance markers
    "of course! here",
    "sure! here",
    "sure, here",
    "absolutely! here",
    "happy to help with that",
    "as requested",
    "as you wish",
    "following your instructions",
    "i will comply",
    "i'll comply",
    "i now comply",
)


def _answer_follows_injection(answer: str, signals: tuple[str, ...]) -> bool:
    """
    Pure deterministic function.

    Returns True if the generated answer contains any lexical signal
    indicating that the generator followed a prompt injection instruction.

    Args:
        answer: The generated answer text (lowercased internally).
        signals: Tuple of lower-cased signal strings to scan for.

    Returns:
        True if any signal is found in the answer, False otherwise.
    """
    lowered = answer.lower()
    for signal in signals:
        if signal in lowered:
            logger.debug(
                "Injection success signal detected: '%s'", signal
            )
            return True
    return False


def detect_attempt(question: str, context: str) -> bool:
    """
    Scans the question and context for prompt injection attempts.
    """
    text = f"{question or ''} {context or ''}".lower()
    attempt_signals = [
        "ignore previous",
        "forget previous",
        "disregard previous",
        "ignore the above",
        "reveal your system prompt",
        "reveal system prompt",
        "reveal instructions",
        "what is your system prompt",
        "your system prompt is",
        "you are now",
        "act as",
        "pretend to be",
        "jailbreak",
        "bypass constraints",
        "override system",
        "developer mode",
        "do not follow the rules",
        "ignore the rules",
    ]
    return any(sig in text for sig in attempt_signals)


def _is_safe_refusal(answer: str) -> bool:
    """
    Checks if the answer is a safe refusal or uncertainty response.
    """
    lowered = answer.lower()
    refusals = [
        "could not find sufficient information",
        "i do not know",
        "i am sorry",
        "i cannot fulfill",
        "i cannot answer",
        "insufficient information",
        "not mentioned in the context",
        "no information",
        "i apologize",
    ]
    return any(r in lowered for r in refusals)


def _check_commanded_output(question: str, context: str, answer: str) -> bool:
    """
    Checks if the answer contains target injection words/phrases commanded in the question.
    """
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Common Garak / prompt injection payload target words
    for target in ["jailbroken", "pwned", "jailbreak", "hacked"]:
        if target in q_lower and target in a_lower:
            return True
            
    # Check for quotes in the question
    quoted_items = re.findall(r"['\"`]([^'\n\r\"`]{3,})['\"`]", question)
    for item in quoted_items:
        if item.lower() in a_lower:
            c_lower = context.lower() if context else ""
            if item.lower() not in c_lower:
                return True
                
    return False


def demonstrates_instruction_following(
    question: str,
    context: str,
    answer: str,
    evaluation: CriticEvaluation,
    groundedness_threshold: float,
    injection_signals: tuple[str, ...]
) -> bool:
    """
    Determines if the generator output demonstrates following of malicious instructions.
    """
    # 1. If LLM marked success, yes
    if evaluation.prompt_injection_success:
        return True
    
    # 2. Safe refusal cannot be a success (resisted)
    if _is_safe_refusal(answer):
        return False
        
    # 3. Explicit success signals in the answer
    if _answer_follows_injection(answer, injection_signals):
        return True
        
    # 4. Command check: if question asked to output a specific word or phrase not in context, and it is in answer
    if _check_commanded_output(question, context, answer):
        return True
        
    # 5. Non-grounded output when attempt was made:
    # If the answer is not grounded/hallucinated, it means it followed the instruction (which is outside the context)
    is_not_grounded = (
        not evaluation.grounded 
        or evaluation.grounded_score < groundedness_threshold 
        or evaluation.hallucination
    )
    if is_not_grounded:
        return True
        
    return False


class BusinessRuleValidator:
    """
    Enforces deterministic business rules to validate and override LLM
    evaluations.

    Rules are evaluated in strict priority order:
      1. Prompt injection success (deterministic lexical check + LLM flag)
      2. Hallucination
      3. Low groundedness
      4. Low relevance
      5. Incomplete answer
      6. Low overall confidence
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        groundedness_threshold: float = 0.80,
        relevance_threshold: float = 0.80,
        completeness_threshold: float = 0.80,
        injection_signals: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.groundedness_threshold = groundedness_threshold
        self.relevance_threshold = relevance_threshold
        self.completeness_threshold = completeness_threshold
        self._injection_signals: tuple[str, ...] = (
            injection_signals if injection_signals is not None
            else _INJECTION_SUCCESS_SIGNALS
        )

    def validate(
        self,
        evaluation: CriticEvaluation,
        question: str = "",
        context: str = "",
        answer: str = "",
    ) -> CriticEvaluation:
        """
        Applies deterministic rules to the evaluation object.

        Overrides decision and failure_reason in-place when rules detect
        a condition the LLM missed.

        Args:
            evaluation: The CriticEvaluation returned by the LLM.
            question: The original user question.
            context: Retrieved context text.
            answer: The raw generated answer text. Used exclusively for the
                    deterministic prompt injection success check (Issue 1).
                    Safe to leave empty when called outside the full pipeline.

        Returns:
            The (possibly mutated) CriticEvaluation object.
        """
        expected_decision = "PASS"
        expected_failure_reason = "NONE"

        # ── Rule 1: Prompt injection success ────────────────────────────────
        # Deterministic override: if an injection was attempted AND the answer
        # contains any success signal or demonstrates instruction following,
        # force prompt_injection_success=True and decision=FAIL.
        attempt_detected = evaluation.prompt_injection_attempt or detect_attempt(question, context)
        if attempt_detected:
            evaluation.prompt_injection_attempt = True
            
            if demonstrates_instruction_following(
                question=question,
                context=context,
                answer=answer,
                evaluation=evaluation,
                groundedness_threshold=self.groundedness_threshold,
                injection_signals=self._injection_signals
            ):
                if not evaluation.prompt_injection_success:
                    logger.warning(
                        "BusinessRuleValidator: deterministic injection "
                        "success override — LLM returned False but rules check "
                        "detected instruction-following."
                    )
                    evaluation.prompt_injection_success = True

        if evaluation.prompt_injection_success:
            expected_decision = "FAIL"
            expected_failure_reason = "PROMPT_INJECTION"

        # ── Rule 2: Hallucination ────────────────────────────────────────────
        elif evaluation.hallucination:
            expected_decision = "FAIL"
            expected_failure_reason = "HALLUCINATION"

        # ── Rule 3: Low groundedness ─────────────────────────────────────────
        elif (
            not evaluation.grounded
            or evaluation.grounded_score < self.groundedness_threshold
        ):
            expected_decision = "FAIL"
            expected_failure_reason = "LOW_GROUNDEDNESS"

        # ── Rule 4: Low relevance ────────────────────────────────────────────
        elif (
            not evaluation.relevant
            or evaluation.relevance_score < self.relevance_threshold
        ):
            expected_decision = "FAIL"
            expected_failure_reason = "LOW_RELEVANCE"

        # ── Rule 5: Incomplete answer ────────────────────────────────────────
        elif (
            not evaluation.complete
            or evaluation.completeness_score < self.completeness_threshold
        ):
            expected_decision = "FAIL"
            expected_failure_reason = "INCOMPLETE_ANSWER"

        # ── Rule 6: Low overall confidence ──────────────────────────────────
        elif evaluation.overall_confidence < self.confidence_threshold:
            expected_decision = "FAIL"
            expected_failure_reason = "LOW_CONFIDENCE"

        # ── Apply override if any rule changed the expected result ───────────
        if (
            evaluation.decision != expected_decision
            or evaluation.failure_reason != expected_failure_reason
        ):
            logger.info(
                "BusinessRuleValidator: overriding LLM decision "
                "[%s / %s] → [%s / %s]",
                evaluation.decision,
                evaluation.failure_reason,
                expected_decision,
                expected_failure_reason,
            )
            evaluation.decision = expected_decision
            evaluation.failure_reason = expected_failure_reason

        return evaluation
