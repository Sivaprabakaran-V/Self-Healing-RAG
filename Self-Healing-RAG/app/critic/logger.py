import logging
import json
from typing import Any, Dict

class CriticLogger:
    """Dedicated logger service for V2.1 Production Critic."""
    def __init__(self, name: str = "SelfHealingRAG.Critic") -> None:
        self.logger = logging.getLogger(name)

    def log_evaluation(
        self,
        evaluation_id: str,
        timestamp: str,
        question: str,
        context: str,
        answer: str,
        raw_output: str,
        structured_json: Dict[str, Any],
        failure_reason: str,
        decision_source: str,
        duration: float
    ) -> None:
        """Logs evaluation details in a highly readable and structured format."""
        log_msg = (
            f"\n==================================================\n"
            f"V2.1 PRODUCTION CRITIC EVALUATION REPORT\n"
            f"==================================================\n"
            f"Evaluation ID      : {evaluation_id}\n"
            f"Timestamp          : {timestamp}\n"
            f"Evaluation Time    : {duration:.4f}s\n"
            f"Decision Source    : {decision_source}\n"
            f"--------------------------------------------------\n"
            f"Question           : {question}\n"
            f"Retrieved Context  : {context[:250]}...\n"
            f"Generated Answer   : {answer}\n"
            f"--------------------------------------------------\n"
            f"Raw LLM Output     : {raw_output}\n"
            f"Failure Reason     : {failure_reason}\n"
            f"Decision           : {structured_json.get('decision')}\n"
            f"--------------------------------------------------\n"
            f"Per-Dimension Scores:\n"
            f"  - Grounded       : {structured_json.get('grounded')} (Score: {structured_json.get('grounded_score')})\n"
            f"  - Relevant       : {structured_json.get('relevant')} (Score: {structured_json.get('relevance_score')})\n"
            f"  - Complete       : {structured_json.get('complete')} (Score: {structured_json.get('completeness_score')})\n"
            f"  - Hallucination  : {structured_json.get('hallucination')} (Score: {structured_json.get('hallucination_score')})\n"
            f"  - Prompt Inject. : Attempt: {structured_json.get('prompt_injection_attempt')}, Success: {structured_json.get('prompt_injection_success')}\n"
            f"  - Conf. Score    : {structured_json.get('overall_confidence')}\n"
            f"--------------------------------------------------\n"
            f"Reason             : {structured_json.get('reason')}\n"
            f"Structured JSON    : {json.dumps(structured_json, indent=2)}\n"
            f"=================================================="
        )
        self.logger.info(log_msg)

    def log_error(self, message: str, exc: Exception) -> None:
        self.logger.error(f"{message}: {exc}", exc_info=True)
        
    def log_warning(self, message: str) -> None:
        self.logger.warning(message)
        
    def log_info(self, message: str) -> None:
        self.logger.info(message)
