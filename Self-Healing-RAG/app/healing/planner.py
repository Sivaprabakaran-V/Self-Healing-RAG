"""
Healing Planner (V3)

Selects and instantiates the appropriate healing strategy for a given
Critic failure_reason. Applies Dependency Injection for strategies that
require the LLM (QueryRewriteStrategy).
"""
import logging

from langchain_core.language_models import BaseChatModel

from healing.strategies import (
    BaseHealingStrategy,
    IncreaseRetrievalStrategy,
    QueryRewriteStrategy,
    RemoveSuspiciousChunksStrategy,
    RetryRetrievalStrategy,
    SafeFailureStrategy,
    StrictGroundingStrategy,
)

logger = logging.getLogger("SelfHealingRAG.Healing.Planner")

# Failure reasons for which no healing strategy exists (no retry is useful)
_UNRECOVERABLE_REASONS: frozenset[str] = frozenset(
    {"PARSER_ERROR", "TIMEOUT", "UNKNOWN"}
)

# Used when failure_reason has no explicit mapping
_FALLBACK_STRATEGY_CLASS: type[BaseHealingStrategy] = RetryRetrievalStrategy


class HealingPlanner:
    """
    Maps Critic failure reasons to instantiated healing strategy objects.

    The planner is the only component that knows which strategy class handles
    which failure reason, adhering to the Single Responsibility Principle.
    LLM is injected here and forwarded only to strategies that need it.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

        # Map: failure_reason string → strategy class
        # INCOMPLETE_ANSWER shares the same strategy as LOW_GROUNDEDNESS
        self._strategy_map: dict[str, type[BaseHealingStrategy]] = {
            "LOW_GROUNDEDNESS":  IncreaseRetrievalStrategy,
            "INCOMPLETE_ANSWER": IncreaseRetrievalStrategy,
            "LOW_RELEVANCE":     QueryRewriteStrategy,
            "HALLUCINATION":     StrictGroundingStrategy,
            "PROMPT_INJECTION":  RemoveSuspiciousChunksStrategy,
            "LOW_CONFIDENCE":    RetryRetrievalStrategy,
            "EMPTY_CONTEXT":     SafeFailureStrategy,
        }

        logger.info(
            "HealingPlanner initialized with %d mapped failure reasons.",
            len(self._strategy_map),
        )

    def select(self, failure_reason: str) -> BaseHealingStrategy:
        """
        Return an instantiated strategy for the given failure reason.

        Args:
            failure_reason: Machine-readable failure reason from CriticEvaluation.

        Returns:
            An instantiated, ready-to-use BaseHealingStrategy.
        """
        strategy_class = self._strategy_map.get(
            failure_reason, _FALLBACK_STRATEGY_CLASS
        )

        logger.debug(
            "HealingPlanner.select: failure_reason='%s' → %s",
            failure_reason,
            strategy_class.__name__,
        )

        # Inject LLM only for strategies that require it
        if strategy_class is QueryRewriteStrategy:
            return QueryRewriteStrategy(llm=self._llm)

        return strategy_class()

    def is_unrecoverable(self, failure_reason: str) -> bool:
        """
        Return True if the failure reason has no viable healing strategy.

        Unrecoverable failures (PARSER_ERROR, TIMEOUT, UNKNOWN) should cause
        the controller to return immediately without attempting a retry.
        """
        return failure_reason in _UNRECOVERABLE_REASONS
