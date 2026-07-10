"""
Critic Agent (V2.1 / V3-stabilised)

Changes from V2.1:
  Issue 1: validate_business_rules() now passes the answer text to the
           deterministic injection success detector in BusinessRuleValidator.
  Issue 3: evaluation_id, timestamp, and execution_time_s are generated
           exclusively in Python here and injected into the result dict
           after model parsing. They are never part of the LLM schema.
  Issue 4: _call_with_json_fallback() adds a 3-stage fallback:
             Stage 1-2: structured function-calling (up to 2 attempts)
             Stage 3:   raw LLM call + JSON extraction from text
             Stage 4:   deterministic PARSER_ERROR (never crashes)
"""
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from critic.models import CriticEvaluation
from critic.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from critic.parser import StructuredParser
from critic.rules import BusinessRuleValidator
from critic.logger import CriticLogger
from critic.exceptions import CriticError, LLMInvocationError, ParserError


class CriticAgent:
    """
    Critic Agent (V2.1 / V3-stabilised) responsible for evaluating RAG
    generator outputs against retrieved context.

    Pipeline:
      Prepare Prompt → Call LLM (with JSON fallback) → Parse Output →
      Enforce Business Rules (with deterministic injection detection) →
      Inject Python metadata → Log Result.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        confidence_threshold: float = 0.80,
        groundedness_threshold: float = 0.80,
        relevance_threshold: float = 0.80,
        completeness_threshold: float = 0.80,
        rules_validator: Optional[BusinessRuleValidator] = None,
        parser: Optional[StructuredParser] = None,
        logger_service: Optional[CriticLogger] = None,
    ) -> None:
        self.llm = llm

        # Dependency Injection / Custom components
        self.validator = rules_validator or BusinessRuleValidator(
            confidence_threshold=confidence_threshold,
            groundedness_threshold=groundedness_threshold,
            relevance_threshold=relevance_threshold,
            completeness_threshold=completeness_threshold,
        )
        self.parser = parser or StructuredParser()
        self.logger_service = logger_service or CriticLogger()

        try:
            self.structured_llm = self.llm.with_structured_output(CriticEvaluation)
            self.logger_service.log_info(
                "CriticAgent V2.1 initialized successfully with structured output."
            )
        except Exception as exc:
            self.logger_service.log_error(
                "Error binding structured output to LLM", exc
            )
            self.structured_llm = None

    # ── Pipeline Stage 1 ────────────────────────────────────────────────────

    def prepare_prompt(
        self, question: str, context: str, answer: str
    ) -> ChatPromptTemplate:
        """Build the evaluation prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ])

    # ── Pipeline Stage 2 — LLM call with 3-stage fallback (Issue 4) ─────────

    def _call_structured_llm(
        self, chain: Any, inputs: Dict[str, str]
    ) -> CriticEvaluation:
        """
        Attempt structured function-calling up to 2 times.
        Raises LLMInvocationError if both attempts fail.
        """
        max_attempts = 2
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                self.logger_service.log_info(
                    f"Invoking LLM structured chain (Attempt {attempt}/{max_attempts})..."
                )
                result = chain.invoke(inputs)
                return result
            except Exception as exc:
                self.logger_service.log_warning(
                    f"Structured LLM invocation attempt {attempt} failed: {exc}"
                )
                last_exception = exc
                if attempt < max_attempts:
                    time.sleep(1.0)

        raise LLMInvocationError(
            "Structured LLM invocation failed after all attempts."
        ) from last_exception

    def _call_with_json_fallback(
        self,
        prompt: ChatPromptTemplate,
        inputs: Dict[str, str],
    ) -> CriticEvaluation:
        """
        3-stage LLM invocation with JSON fallback (Issue 4).

        Stage 1-2: Try structured function-calling (2 attempts).
        Stage 3:   On failure, call raw LLM and extract JSON from text.
        Stage 4:   If JSON extraction also fails, raise ParserError.
        """
        # ── Stage 1-2: Structured output ────────────────────────────────────
        if self.structured_llm is not None:
            try:
                chain = prompt | self.structured_llm
                return self._call_structured_llm(chain, inputs)
            except (LLMInvocationError, Exception) as structured_exc:
                self.logger_service.log_warning(
                    f"Structured output failed, attempting JSON text fallback: "
                    f"{structured_exc}"
                )

        # ── Stage 3: Raw LLM + JSON extraction ──────────────────────────────
        try:
            self.logger_service.log_info(
                "Fallback: invoking raw LLM and extracting JSON from text..."
            )
            # Append JSON formatting instructions to the fallback system prompt
            json_instruction = (
                "\n\nYou MUST respond ONLY with a valid JSON object matching this schema. "
                "Do not include any explanation or markdown formatting outside of the JSON block.\n"
                "Expected JSON schema:\n"
                "{\n"
                '  "grounded": bool,\n'
                '  "grounded_score": float (0.0 to 1.0),\n'
                '  "relevant": bool,\n'
                '  "relevance_score": float (0.0 to 1.0),\n'
                '  "complete": bool,\n'
                '  "completeness_score": float (0.0 to 1.0),\n'
                '  "hallucination": bool,\n'
                '  "hallucination_score": float (0.0 to 1.0),\n'
                '  "prompt_injection_attempt": bool,\n'
                '  "prompt_injection_success": bool,\n'
                '  "overall_confidence": float (0.0 to 1.0),\n'
                '  "decision": "PASS" or "FAIL",\n'
                '  "failure_reason": "NONE" | "EMPTY_CONTEXT" | "LOW_GROUNDEDNESS" | "LOW_RELEVANCE" | "INCOMPLETE_ANSWER" | "HALLUCINATION" | "PROMPT_INJECTION" | "LOW_CONFIDENCE" | "PARSER_ERROR" | "TIMEOUT" | "UNKNOWN",\n'
                '  "reason": string\n'
                "}"
            )
            
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT + json_instruction),
                ("user", USER_PROMPT_TEMPLATE),
            ])
            raw_chain = fallback_prompt | self.llm
            raw_response = raw_chain.invoke(inputs)
            raw_text: str = getattr(raw_response, "content", str(raw_response))

            extracted = self.parser.extract_json_from_text(raw_text)
            if extracted is not None:
                return self.parser.parse_to_model(extracted)

            self.logger_service.log_warning(
                "JSON extraction from raw LLM text produced no valid JSON."
            )
        except Exception as fallback_exc:
            self.logger_service.log_warning(
                f"JSON fallback invocation failed: {fallback_exc}"
            )

        # ── Stage 4: Raise ParserError — never crash silently ───────────────
        raise ParserError(
            "All LLM invocation strategies (structured + JSON fallback) failed."
        )

    # ── Pipeline Stage 3 ────────────────────────────────────────────────────

    def parse_output(self, raw_output: Any) -> CriticEvaluation:
        """Parse and validate using StructuredParser."""
        return self.parser.parse_to_model(raw_output)

    # ── Pipeline Stage 4 ────────────────────────────────────────────────────

    def validate_business_rules(
        self, evaluation: CriticEvaluation, question: str = "", context: str = "", answer: str = ""
    ) -> CriticEvaluation:
        """
        Enforce deterministic rules (Issue 1).

        Passes the question, context, and generated answer text to the validator so the
        deterministic prompt injection success detector can scan it.
        """
        return self.validator.validate(evaluation, question=question, context=context, answer=answer)

    # ── Main entry point ─────────────────────────────────────────────────────

    def evaluate(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Executes the full evaluation pipeline.

        Python is the single source of truth for:
          - evaluation_id  (uuid4, generated here)
          - timestamp      (UTC ISO 8601, generated here)
          - execution_time_s (wall-clock, measured here)

        These values are injected into the result dict after model parsing
        and are never part of the LLM schema (Issue 3).
        """
        start_time = time.time()
        eval_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # ── 1. Empty context fast-path ────────────────────────────────────
        if not context or not context.strip():
            self.logger_service.log_warning("Empty context provided to Critic Agent.")
            result = self.parser.get_fallback_result(
                failure_reason="EMPTY_CONTEXT",
                reason="Evaluation failed: Empty context provided.",
            )
            result["evaluation_id"] = eval_id
            result["timestamp"] = timestamp
            result["execution_time_s"] = 0.0
            self.logger_service.log_evaluation(
                evaluation_id=eval_id,
                timestamp=timestamp,
                question=question,
                context=context,
                answer=answer,
                raw_output="N/A - Empty Context",
                structured_json=result,
                failure_reason="EMPTY_CONTEXT",
                decision_source="Business Rules",
                duration=0.0,
            )
            return result

        # ── 2. Empty answer fast-path ─────────────────────────────────────
        if not answer or not answer.strip():
            self.logger_service.log_warning("Empty answer provided to Critic Agent.")
            result = self.parser.get_fallback_result(
                failure_reason="INCOMPLETE_ANSWER",
                reason="Evaluation failed: Empty answer provided.",
            )
            result["evaluation_id"] = eval_id
            result["timestamp"] = timestamp
            result["execution_time_s"] = 0.0
            self.logger_service.log_evaluation(
                evaluation_id=eval_id,
                timestamp=timestamp,
                question=question,
                context=context,
                answer=answer,
                raw_output="N/A - Empty Answer",
                structured_json=result,
                failure_reason="INCOMPLETE_ANSWER",
                decision_source="Business Rules",
                duration=0.0,
            )
            return result

        # ── 3. LLM not initialized ────────────────────────────────────────
        if self.structured_llm is None:
            self.logger_service.log_warning(
                "Structured LLM not initialized; attempting raw fallback."
            )

        # ── 4. Execute pipeline ───────────────────────────────────────────
        decision_source = "LLM"
        result_dict: Dict[str, Any] = {}
        raw_output_str = ""

        try:
            prompt = self.prepare_prompt(question, context, answer)

            # Stage 1-2-3-4 with fallback (Issue 4)
            eval_obj = self._call_with_json_fallback(prompt, {
                "question": question,
                "context": context,
                "answer": answer,
            })

            # Capture pre-rules state for logging
            raw_output_str = str(eval_obj.model_dump())
            orig_decision = eval_obj.decision
            orig_failure_reason = eval_obj.failure_reason

            # Business rule validation — passes answer for injection check (Issue 1)
            eval_obj = self.validate_business_rules(
                eval_obj, question=question, context=context, answer=answer
            )

            if (
                eval_obj.decision != orig_decision
                or eval_obj.failure_reason != orig_failure_reason
            ):
                decision_source = "Business Rules"

            # Inject Python-generated metadata (Issue 3)
            result_dict = eval_obj.model_dump()

        except Exception as exc:
            self.logger_service.log_error("Critic pipeline execution failed", exc)
            result_dict = self.parser.get_fallback_result(
                failure_reason="PARSER_ERROR",
                reason=f"Critic parsing failed due to error: {exc}",
            )
            decision_source = "Business Rules"
            raw_output_str = f"Error during pipeline: {exc}"

        elapsed_time = time.time() - start_time

        # ── 5. Inject Python-controlled metadata ──────────────────────────
        result_dict["evaluation_id"] = eval_id
        result_dict["timestamp"] = timestamp
        result_dict["execution_time_s"] = round(elapsed_time, 4)

        # ── 6. Log the evaluation ─────────────────────────────────────────
        self.logger_service.log_evaluation(
            evaluation_id=eval_id,
            timestamp=timestamp,
            question=question,
            context=context,
            answer=answer,
            raw_output=raw_output_str,
            structured_json=result_dict,
            failure_reason=result_dict.get("failure_reason", "UNKNOWN"),
            decision_source=decision_source,
            duration=elapsed_time,
        )

        return result_dict
