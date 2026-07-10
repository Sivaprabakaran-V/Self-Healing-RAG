"""
Retry Orchestrator (V3)

Manages the healing retry loop:

    for attempt in range(max_retries):
        strategy.apply(context)          ← mutate retrieval params
        _retrieve_with_context(context)  ← respect k, exclusions, rewrite
        _generate_with_context(...)      ← optional strict grounding
        CriticAgent.evaluate()           ← always fresh evaluation
        if PASS → return immediately
        else    → re-select strategy from new failure_reason

Max retries is 2. The Critic is called on every single retry.
"""
import logging
import time
from typing import TYPE_CHECKING, Any

from healing.models import HealingAttempt, HealingStrategy, RetryContext
from healing.planner import HealingPlanner
from healing.strategies import SafeFailureSignal

if TYPE_CHECKING:
    # Avoid circular import at runtime; rag.py imports HealingController
    from rag import SelfHealingRAG

logger = logging.getLogger("SelfHealingRAG.Healing.Retry")

_SAFE_FAILURE_ANSWER = (
    "I could not find sufficient information in the uploaded documents."
)


# ---------------------------------------------------------------------------
# Structured Healing Logger
# ---------------------------------------------------------------------------

class HealingLogger:
    """Structured per-attempt logger for the Healing Controller."""

    def __init__(self) -> None:
        self._log = logging.getLogger("SelfHealingRAG.Healing")

    def log_attempt(
        self,
        *,
        evaluation_id: str,
        retry_number: int,
        max_retries: int,
        strategy_name: str,
        failure_reason: str,
        retrieval_k: int,
        outcome: str,
        execution_time_s: float,
    ) -> None:
        msg = (
            f"\n{'='*52}\n"
            f"HEALING CONTROLLER — ATTEMPT {retry_number}/{max_retries}\n"
            f"{'='*52}\n"
            f"  Evaluation ID   : {evaluation_id}\n"
            f"  Retry Number    : {retry_number}\n"
            f"  Strategy        : {strategy_name}\n"
            f"  Failure Reason  : {failure_reason}\n"
            f"  Retrieval k     : {retrieval_k}\n"
            f"  Outcome         : {outcome}\n"
            f"  Execution Time  : {execution_time_s:.4f}s\n"
            f"{'='*52}"
        )
        self._log.info(msg)

    def log_safe_failure(self, reason: str) -> None:
        self._log.warning("HealingController: SAFE_FAILURE — %s", reason)

    def log_exhausted(self, retry_count: int) -> None:
        self._log.warning(
            "HealingController: retries exhausted after %d attempt(s).",
            retry_count,
        )

    def log_unrecoverable(self, failure_reason: str) -> None:
        self._log.error(
            "HealingController: unrecoverable failure_reason='%s'. "
            "Skipping retry loop.",
            failure_reason,
        )


# ---------------------------------------------------------------------------
# Retry Orchestrator
# ---------------------------------------------------------------------------

class RetryOrchestrator:
    """
    Executes the retry loop: strategy → retrieve → generate → critic.

    Depends on SelfHealingRAG internal methods:
      - _retrieve_with_context(RetryContext) -> dict
      - _generate_with_context(question, RetryContext, context_text) -> str

    These are thin wrappers around the existing retrieval/generation code
    that accept RetryContext parameters. The public retrieve_context() and
    generate_answer() methods on SelfHealingRAG are never modified.
    """

    def __init__(self, rag: "SelfHealingRAG", max_retries: int = 2) -> None:
        self._rag = rag
        self._max_retries = max_retries
        self._healing_logger = HealingLogger()

    def execute(
        self,
        context: RetryContext,
        failure_reason: str,
        planner: HealingPlanner,
    ) -> dict[str, Any]:
        """
        Run the full retry loop.

        Args:
            context: Initial RetryContext populated by the HealingController.
            failure_reason: The failure_reason from the initial Critic FAIL.
            planner: HealingPlanner instance for strategy selection.

        Returns:
            A dict with keys: answer, critic_evaluation, sources,
            retrieved_chunks, attempts, status.
        """
        attempts: list[HealingAttempt] = []
        current_failure_reason = failure_reason
        last_critic_result: dict[str, Any] = {}
        last_answer: str = _SAFE_FAILURE_ANSWER
        last_sources: list[str] = []
        last_chunks: int = 0

        for attempt_num in range(1, self._max_retries + 1):
            t_start = time.time()

            # ── 1. Select strategy ─────────────────────────────────────────
            strategy = planner.select(current_failure_reason)
            logger.info(f"[Self-Healing RAG] Retry {attempt_num}:")
            logger.info(f"[Self-Healing RAG]   Selected Strategy: {strategy.name}")

            # ── 2. Apply strategy (mutates context) ────────────────────────
            try:
                context = strategy.apply(context)
                logger.info(
                    f"[Self-Healing RAG]   Retrieval Parameters: k={context.retrieval_k}, fetch_k={context.fetch_k}, "
                    f"query='{context.rewritten_query or context.question}', excluded={len(context.excluded_chunk_ids)}"
                )
            except SafeFailureSignal as sig:
                elapsed = time.time() - t_start
                self._healing_logger.log_safe_failure(str(sig))
                logger.info(f"[Self-Healing RAG]   Safe Failure triggered: {sig}")
                attempts.append(
                    HealingAttempt(
                        attempt_number=attempt_num,
                        strategy=HealingStrategy(strategy.name),
                        failure_reason=current_failure_reason,
                        retrieval_k=context.retrieval_k,
                        outcome="SAFE_FAILURE",
                        execution_time_s=elapsed,
                    )
                )
                return {
                    "answer": _SAFE_FAILURE_ANSWER,
                    "critic_evaluation": last_critic_result,
                    "sources": [],
                    "retrieved_chunks": 0,
                    "attempts": attempts,
                    "status": "FAIL",
                }

            # ── 3. Retrieve ────────────────────────────────────────────────
            retrieval_res = self._rag._retrieve_with_context(context)
            docs = retrieval_res["documents"]
            scores = retrieval_res["relevance_scores"]

            # Update last_retrieved_chunk_ids so next strategy can reference them
            context.last_retrieved_chunk_ids = [
                doc.metadata.get("chunk_id", "")
                for doc in docs
                if doc.metadata.get("chunk_id")
            ]

            # ── 4. Filter by relevance threshold ───────────────────────────
            filtered_docs = [
                doc for doc, score in zip(docs, scores)
                if score >= self._rag.min_relevance_score
            ]
            filtered_sources = list({
                doc.metadata.get("source_file", "")
                for doc in filtered_docs
                if doc.metadata.get("source_file")
            })
            context_text = "\n\n".join(
                doc.page_content for doc in filtered_docs
            )
            logger.info(f"[Self-Healing RAG]   Retrieval: Retrieved {len(filtered_docs)} chunks")

            # ── 5. Generate ────────────────────────────────────────────────
            if not filtered_docs or not context_text.strip():
                answer = _SAFE_FAILURE_ANSWER
                context_text = ""
                logger.info("[Self-Healing RAG]   Generation: Empty context, using safe failure answer")
            else:
                answer = self._rag._generate_with_context(
                    context.question, context, context_text
                )
                logger.info("[Self-Healing RAG]   Generation: Regenerated answer")

            # ── 6. Critic — always a fresh evaluation ──────────────────────
            # Issue 2: Always evaluate against the original question, not rewritten
            critic_result = self._rag.critic.evaluate(
                context.question, context_text, answer
            )

            elapsed = time.time() - t_start
            outcome = critic_result.get("decision", "FAIL")
            current_failure_reason = critic_result.get("failure_reason", "UNKNOWN")
            
            logger.info(f"[Self-Healing RAG]   Critic evaluation: {outcome} (Reason: {current_failure_reason})")
            logger.info(f"[Self-Healing RAG]   Execution Time: {elapsed:.4f}s")

            # ── 7. Log the attempt ─────────────────────────────────────────
            self._healing_logger.log_attempt(
                evaluation_id=critic_result.get("evaluation_id", "N/A"),
                retry_number=attempt_num,
                max_retries=self._max_retries,
                strategy_name=strategy.name,
                failure_reason=critic_result.get("failure_reason", "UNKNOWN"),
                retrieval_k=context.retrieval_k,
                outcome=outcome,
                execution_time_s=elapsed,
            )

            attempts.append(
                HealingAttempt(
                    attempt_number=attempt_num,
                    strategy=HealingStrategy(strategy.name),
                    failure_reason=current_failure_reason,
                    retrieval_k=context.retrieval_k,
                    outcome=outcome,
                    execution_time_s=elapsed,
                )
            )

            # ── 8. Persist best result so far ──────────────────────────────
            last_critic_result = critic_result
            last_answer = answer
            last_sources = filtered_sources
            last_chunks = len(filtered_docs)

            if outcome == "PASS":
                return {
                    "answer": answer,
                    "critic_evaluation": critic_result,
                    "sources": sorted(last_sources),
                    "retrieved_chunks": last_chunks,
                    "attempts": attempts,
                    "status": "HEALED",
                }

            # Prepare failure_reason for next iteration
            current_failure_reason = critic_result.get(
                "failure_reason", "UNKNOWN"
            )

            # Stop early if unrecoverable
            if planner.is_unrecoverable(current_failure_reason):
                logger.warning(
                    "RetryOrchestrator: unrecoverable failure after attempt %d.",
                    attempt_num,
                )
                break

        # ── Retries exhausted ─────────────────────────────────────────────
        self._healing_logger.log_exhausted(len(attempts))
        return {
            "answer": last_answer,
            "critic_evaluation": last_critic_result,
            "sources": sorted(last_sources),
            "retrieved_chunks": last_chunks,
            "attempts": attempts,
            "status": "FAIL",
        }
