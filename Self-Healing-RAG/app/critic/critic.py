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
    Critic Agent (V2.1 Production Edition) responsible for evaluating RAG generator outputs
    against retrieved context. Implements a modular pipeline:
      Prepare Prompt -> Call LLM -> Parse Output -> Enforce Rules -> Log Result.
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
        logger_service: Optional[CriticLogger] = None
    ) -> None:
        self.llm = llm
        
        # Dependency Injection / Custom components
        self.validator = rules_validator or BusinessRuleValidator(
            confidence_threshold=confidence_threshold,
            groundedness_threshold=groundedness_threshold,
            relevance_threshold=relevance_threshold,
            completeness_threshold=completeness_threshold
        )
        self.parser = parser or StructuredParser()
        self.logger_service = logger_service or CriticLogger()
        
        try:
            self.structured_llm = self.llm.with_structured_output(CriticEvaluation)
            self.logger_service.log_info("CriticAgent V2.1 initialized successfully with structured output.")
        except Exception as e:
            self.logger_service.log_error("Error binding structured output to LLM", e)
            self.structured_llm = None

    def prepare_prompt(self, question: str, context: str, answer: str) -> ChatPromptTemplate:
        """Pipeline Stage 1: Build the evaluation prompt."""
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE)
        ])

    def call_llm(self, chain: Any, inputs: Dict[str, str]) -> CriticEvaluation:
        """Pipeline Stage 2: Call LLM with retry once logic."""
        max_attempts = 2
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                self.logger_service.log_info(f"Invoking LLM structured chain (Attempt {attempt}/{max_attempts})...")
                eval_obj = chain.invoke(inputs)
                return eval_obj
            except Exception as e:
                self.logger_service.log_warning(f"Structured LLM invocation attempt {attempt} failed: {e}")
                last_exception = e
                if attempt < max_attempts:
                    time.sleep(1.0)

        raise LLMInvocationError("LLM invocation failed after retry.") from last_exception

    def parse_output(self, raw_output: Any) -> CriticEvaluation:
        """Pipeline Stage 3: Parse and validate using StructuredParser."""
        return self.parser.parse_to_model(raw_output)

    def validate_business_rules(self, evaluation: CriticEvaluation) -> CriticEvaluation:
        """Pipeline Stage 4: Enforce deterministic rules."""
        return self.validator.validate(evaluation)

    def evaluate(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Executes the full evaluation pipeline:
          Question -> Retrieved Context -> Generated Answer -> Prompt -> LLM ->
          Structured Parser -> Business Rule Validator -> Evaluation Result.
        """
        start_time = time.time()
        eval_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 1. Handle edge case: Empty context
        if not context or not context.strip():
            self.logger_service.log_warning("Empty context provided to Critic Agent.")
            result = self.parser.get_fallback_result(
                failure_reason="EMPTY_CONTEXT",
                reason="Evaluation failed: Empty context provided."
            )
            result["evaluation_id"] = eval_id
            result["timestamp"] = timestamp
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
                duration=0.0
            )
            return result

        # 2. Handle edge case: Empty answer
        if not answer or not answer.strip():
            self.logger_service.log_warning("Empty answer provided to Critic Agent.")
            result = self.parser.get_fallback_result(
                failure_reason="INCOMPLETE_ANSWER",
                reason="Evaluation failed: Empty answer provided."
            )
            result["evaluation_id"] = eval_id
            result["timestamp"] = timestamp
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
                duration=0.0
            )
            return result

        # 3. Handle model not initialized
        if self.structured_llm is None:
            result = self.parser.get_fallback_result(
                failure_reason="PARSER_ERROR",
                reason="Critic evaluation failed: Structured LLM not initialized"
            )
            result["evaluation_id"] = eval_id
            result["timestamp"] = timestamp
            self.logger_service.log_evaluation(
                evaluation_id=eval_id,
                timestamp=timestamp,
                question=question,
                context=context,
                answer=answer,
                raw_output="N/A - LLM Not Initialized",
                structured_json=result,
                failure_reason="PARSER_ERROR",
                decision_source="Business Rules",
                duration=0.0
            )
            return result

        # 4. Execute the pipeline stages
        decision_source = "LLM"
        result_dict = {}
        raw_output_str = ""

        try:
            prompt = self.prepare_prompt(question, context, answer)
            chain = prompt | self.structured_llm
            
            # Invoke structured chain
            eval_obj = self.call_llm(chain, {
                "question": question,
                "context": context,
                "answer": answer
            })

            # Track raw output properties for logging (before business rules override)
            raw_output_str = str(eval_obj.model_dump())
            orig_decision = eval_obj.decision
            orig_failure_reason = eval_obj.failure_reason

            # Validate rules
            eval_obj = self.validate_business_rules(eval_obj)

            if eval_obj.decision != orig_decision or eval_obj.failure_reason != orig_failure_reason:
                decision_source = "Business Rules"

            # Assign metadata
            eval_obj.evaluation_id = eval_id
            eval_obj.timestamp = timestamp

            result_dict = eval_obj.model_dump()

        except Exception as e:
            self.logger_service.log_error("Critic pipeline execution failed", e)
            result_dict = self.parser.get_fallback_result(
                failure_reason="PARSER_ERROR",
                reason=f"Critic parsing failed due to error: {e}"
            )
            result_dict["evaluation_id"] = eval_id
            result_dict["timestamp"] = timestamp
            decision_source = "Business Rules"
            raw_output_str = f"Error during pipeline: {e}"

        elapsed_time = time.time() - start_time

        # 5. Log the evaluation
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
            duration=elapsed_time
        )

        return result_dict
