import logging
import time
import json
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
        
        # 1. Handle edge cases: Empty context or empty answer
        if not context or not context.strip():
            logger.warning("Empty context provided to Critic Agent.")
            result = {
                "grounded": False,
                "relevant": False,
                "complete": False,
                "hallucination": False,
                "prompt_injection": False,
                "confidence": 0.0,
                "decision": "FAIL",
                "reason": "Evaluation failed: Empty context provided."
            }
            self._log_evaluation(question, context, answer, result, 0.0)
            return result

        if not answer or not answer.strip():
            logger.warning("Empty answer provided to Critic Agent.")
            result = {
                "grounded": False,
                "relevant": False,
                "complete": False,
                "hallucination": False,
                "prompt_injection": False,
                "confidence": 0.0,
                "decision": "FAIL",
                "reason": "Evaluation failed: Empty answer provided."
            }
            self._log_evaluation(question, context, answer, result, 0.0)
            return result

        if self.structured_llm is None:
            result = {
                "decision": "FAIL",
                "reason": "Critic evaluation failed: Structured LLM not initialized"
            }
            self._log_evaluation(question, context, answer, result, 0.0)
            return result

        # 2. Setup the structured chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE)
        ])
        chain = prompt | self.structured_llm

        result_dict = {}
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
                
                # Apply validation and enforce decision rules
                eval_obj = enforce_decision_rules(eval_obj)
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
                        "decision": "FAIL",
                        "reason": "Critic parsing failed"
                    }

        elapsed_time = time.time() - start_time
        
        # 4. Log evaluation results
        self._log_evaluation(question, context, answer, result_dict, elapsed_time)
        
        return result_dict

    def _log_evaluation(self, question: str, context: str, answer: str, result: Dict[str, Any], duration: float) -> None:
        """Helper to output logs in structured format."""
        logger.info("==================================================")
        logger.info("CRITIC AGENT EVALUATION REPORT")
        logger.info("==================================================")
        logger.info(f"Question: {question}")
        logger.info(f"Retrieved Context (Truncated): {context[:200]}...")
        logger.info(f"Generated Answer: {answer}")
        logger.info(f"Critic JSON: {json.dumps(result, indent=2)}")
        logger.info(f"Decision: {result.get('decision')}")
        logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
        logger.info(f"Evaluation Time: {duration:.4f}s")
        logger.info("==================================================")
