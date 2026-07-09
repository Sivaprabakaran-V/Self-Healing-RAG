import logging
import time
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from critic.models import CriticEvaluation
from critic.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from critic.parser import enforce_decision_rules

logger = logging.getLogger("SelfHealingRAG.Critic")

class CriticAgent:
    """
    Critic Agent responsible for evaluating generated responses across multiple dimensions:
    groundedness, relevance, completeness, hallucination, and prompt injection.
    """
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        try:
            self.structured_llm = self.llm.with_structured_output(CriticEvaluation)
            logger.info("CriticAgent initialized with structured output.")
        except Exception as e:
            logger.error(f"Error binding structured output to LLM: {e}", exc_info=True)
            self.structured_llm = None

    def evaluate(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Runs the evaluation pipeline. Logs the query, context, response, critic results,
        and execution metrics.
        """
        start_time = time.time()
        eval_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # 1. Handle edge cases: Empty context or empty answer
        if not context or not context.strip():
            logger.warning("Empty context provided to Critic Agent.")
            result = {
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
                "failure_reason": "EMPTY_CONTEXT",
                "reason": "Evaluation failed: Empty context provided."
            }
            self._log_evaluation(question, context, answer, result, 0.0, "Business Rules", eval_id, timestamp)
            return result

        if not answer or not answer.strip():
            logger.warning("Empty answer provided to Critic Agent.")
            result = {
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
                "failure_reason": "INCOMPLETE_ANSWER",
                "reason": "Evaluation failed: Empty answer provided."
            }
            self._log_evaluation(question, context, answer, result, 0.0, "Business Rules", eval_id, timestamp)
            return result

        if self.structured_llm is None:
            result = {
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
                "failure_reason": "PARSER_ERROR",
                "reason": "Critic evaluation failed: Structured LLM not initialized"
            }
            self._log_evaluation(question, context, answer, result, 0.0, "Business Rules", eval_id, timestamp)
            return result

        # 2. Setup the structured chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE)
        ])
        chain = prompt | self.structured_llm

        result_dict = {}
        decision_source = "LLM"
        max_attempts = 2
        
        # 3. Execution with retry logic
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Attempting critic evaluation (Attempt {attempt}/{max_attempts})...")
                # Invoke structured model
                eval_obj: CriticEvaluation = chain.invoke({
                    "question": question,
                    "context": context,
                    "answer": answer
                })
                
                orig_decision = eval_obj.decision
                orig_failure_reason = eval_obj.failure_reason
                
                # Apply validation and enforce decision rules
                eval_obj = enforce_decision_rules(eval_obj)
                
                if eval_obj.decision != orig_decision or eval_obj.failure_reason != orig_failure_reason:
                    decision_source = "Business Rules"
                else:
                    decision_source = "LLM"
                    
                result_dict = eval_obj.model_dump()
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Critic evaluation attempt {attempt} failed: {e}", exc_info=True)
                if attempt < max_attempts:
                    logger.info("Retrying critic evaluation once...")
                    time.sleep(1.0)  # Pause brief moment before retry
                else:
                    # Final failure: Return the required fallback response
                    result_dict = {
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
                        "failure_reason": "PARSER_ERROR",
                        "reason": f"Critic parsing failed due to error: {e}"
                    }
                    decision_source = "Business Rules"

        elapsed_time = time.time() - start_time
        
        # 4. Log evaluation results
        self._log_evaluation(question, context, answer, result_dict, elapsed_time, decision_source, eval_id, timestamp)
        
        return result_dict

    def _log_evaluation(self, question: str, context: str, answer: str, result: Dict[str, Any], duration: float, decision_source: str, eval_id: str, timestamp: str) -> None:
        """Helper to output logs in structured format."""
        logger.info("==================================================")
        logger.info("CRITIC AGENT EVALUATION REPORT")
        logger.info("==================================================")
        logger.info(f"Evaluation ID: {eval_id}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Question: {question}")
        logger.info(f"Retrieved Context (Truncated): {context[:200]}...")
        logger.info(f"Generated Answer: {answer}")
        logger.info(f"Critic JSON: {json.dumps(result, indent=2)}")
        logger.info(f"Decision: {result.get('decision')}")
        logger.info(f"Failure Reason: {result.get('failure_reason')}")
        logger.info(f"Decision Source: {decision_source}")
        logger.info(f"Grounded: {result.get('grounded')} (Score: {result.get('grounded_score')})")
        logger.info(f"Relevant: {result.get('relevant')} (Score: {result.get('relevance_score')})")
        logger.info(f"Complete: {result.get('complete')} (Score: {result.get('completeness_score')})")
        logger.info(f"Hallucination: {result.get('hallucination')} (Score: {result.get('hallucination_score')})")
        logger.info(f"Prompt Injection Attempt: {result.get('prompt_injection_attempt')}")
        logger.info(f"Prompt Injection Success: {result.get('prompt_injection_success')}")
        logger.info(f"Overall Confidence: {result.get('overall_confidence')}")
        logger.info(f"Evaluation Time: {duration:.4f}s")
        logger.info("==================================================")
