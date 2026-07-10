"""
Healing Controller (V3) — Top-Level Entry Point

The HealingController is the single public interface for the healing layer.
It is called by SelfHealingRAG.ask() exclusively when the CriticAgent
returns decision='FAIL'.

Responsibilities:
  - Build the initial RetryContext from the first Critic result
  - Skip healing for unrecoverable failure reasons
  - Delegate to RetryOrchestrator for all retry logic
  - Return a fully populated HealingResult

The controller NEVER evaluates responses. Evaluation belongs to the Critic.
"""
import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel

from healing.models import HealingAttempt, HealingResult, HealingStrategy, RetryContext
from healing.planner import HealingPlanner
from healing.retry import RetryOrchestrator

if TYPE_CHECKING:
    from rag import SelfHealingRAG

logger = logging.getLogger("SelfHealingRAG.Healing.Controller")

_SAFE_FAILURE_ANSWER = (
    "I could not find sufficient information in the uploaded documents."
)


class HealingController:
    """
    Top-level orchestrator for the Healing Controller layer (V3).

    Composed of:
      - HealingPlanner: maps failure_reason → strategy class
      - RetryOrchestrator: drives the retry loop

    Usage (from SelfHealingRAG.ask()):
        result = self.healing_controller.run(
            question=...,
            initial_critic_result=critic_res,
            initial_answer=answer,
            initial_sources=sources,
            initial_chunks=len(filtered_docs),
            initial_docs=filtered_docs,
        )
    """

    def __init__(
        self,
        rag: "SelfHealingRAG",
        llm: BaseChatModel,
        max_retries: int = 2,
    ) -> None:
        self._rag = rag
        self._max_retries = max_retries
        self._planner = HealingPlanner(llm=llm)
        self._orchestrator = RetryOrchestrator(rag=rag, max_retries=max_retries)
        logger.info(
            "HealingController V3 initialized (max_retries=%d).", max_retries
        )

    def run(
        self,
        question: str,
        initial_critic_result: dict[str, Any],
        initial_answer: str,
        initial_sources: list[str],
        initial_chunks: int,
        initial_docs: list[Any],
    ) -> HealingResult:
        """
        Execute the healing flow for a Critic FAIL result.

        Args:
            question: The original user question.
            initial_critic_result: The CriticAgent evaluation dict that returned FAIL.
            initial_answer: The answer generated before healing.
            initial_sources: Source files from the initial retrieval.
            initial_chunks: Number of context chunks from the initial retrieval.
            initial_docs: Document objects from the initial retrieval (used to
                          populate last_retrieved_chunk_ids for PROMPT_INJECTION).

        Returns:
            A HealingResult with the best answer, final critic evaluation,
            all attempt records, and execution metadata.
        """
        healing_start = time.time()

        failure_reason = initial_critic_result.get("failure_reason", "UNKNOWN")
        evaluation_id = initial_critic_result.get("evaluation_id", "N/A")

        logger.info(
            "HealingController.run: evaluation_id=%s failure_reason='%s'",
            evaluation_id,
            failure_reason,
        )

        # ── Fast-path: unrecoverable failure ──────────────────────────────
        if self._planner.is_unrecoverable(failure_reason):
            logger.error(
                "HealingController: failure_reason='%s' is unrecoverable. "
                "Returning initial answer without healing.",
                failure_reason,
            )
            return HealingResult(
                healing_attempted=False,
                final_strategy=None,
                healing_strategy=None,
                retry_count=0,
                attempts=[],
                final_answer=initial_answer,
                status="FAIL",
                final_critic_evaluation=initial_critic_result,
                sources=initial_sources,
                retrieved_chunks=initial_chunks,
                total_healing_time_s=time.time() - healing_start,
                initial_failure_reason=failure_reason,
                healing_success=False,
                final_decision=initial_critic_result.get("decision", "FAIL"),
            )

        # ── Build initial RetryContext ─────────────────────────────────────
        initial_chunk_ids = [
            doc.metadata.get("chunk_id", "")
            for doc in initial_docs
            if doc.metadata.get("chunk_id")
        ]
        context = RetryContext(
            question=question,
            retrieval_k=5,       # Default; strategies may increase this
            fetch_k=20,
            last_retrieved_chunk_ids=initial_chunk_ids,
        )

        # ── Delegate to RetryOrchestrator ─────────────────────────────────
        retry_result = self._orchestrator.execute(
            context=context,
            failure_reason=failure_reason,
            planner=self._planner,
        )

        total_healing_time = time.time() - healing_start
        attempts: list[HealingAttempt] = retry_result["attempts"]

        # Determine final strategy from last attempt
        final_strategy: HealingStrategy | None = None
        if attempts:
            final_strategy = attempts[-1].strategy

        resolved_status = retry_result["status"]
        return HealingResult(
            healing_attempted=True,
            final_strategy=final_strategy,
            healing_strategy=final_strategy,
            retry_count=len(attempts),
            attempts=attempts,
            final_answer=retry_result["answer"],
            status=resolved_status,
            final_critic_evaluation=retry_result["critic_evaluation"],
            sources=retry_result["sources"],
            retrieved_chunks=retry_result["retrieved_chunks"],
            total_healing_time_s=total_healing_time,
            initial_failure_reason=failure_reason,
            healing_success=(resolved_status == "HEALED"),
            final_decision=retry_result["critic_evaluation"].get("decision", "FAIL"),
        )
