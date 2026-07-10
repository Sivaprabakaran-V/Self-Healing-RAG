"""
Healing Strategies (V3) — Strategy Pattern Implementation

Each class encapsulates exactly one recovery action applied to a RetryContext.
Strategies NEVER call retrieval, generation, or the Critic. They only mutate
the RetryContext fields that drive the next retry cycle.

Strategy → Failure Reason mapping
──────────────────────────────────────────────────────────────
IncreaseRetrievalStrategy       LOW_GROUNDEDNESS, INCOMPLETE_ANSWER
QueryRewriteStrategy            LOW_RELEVANCE
StrictGroundingStrategy         HALLUCINATION
RemoveSuspiciousChunksStrategy  PROMPT_INJECTION
RetryRetrievalStrategy          LOW_CONFIDENCE
SafeFailureStrategy             EMPTY_CONTEXT
"""
import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from healing.models import RetryContext

logger = logging.getLogger("SelfHealingRAG.Healing.Strategies")

# Retrieval cap — prevents runaway API costs on successive retries
_MAX_K: int = 15
_K_INCREMENT: int = 3


# ---------------------------------------------------------------------------
# Signal exception (not an error — used for control flow only)
# ---------------------------------------------------------------------------

class SafeFailureSignal(Exception):
    """
    Raised by SafeFailureStrategy to immediately short-circuit the retry loop
    and return a predefined safe refusal message. Not a system error.
    """


# ---------------------------------------------------------------------------
# Abstract Base Strategy
# ---------------------------------------------------------------------------

class BaseHealingStrategy(ABC):
    """
    Abstract base class enforcing the Strategy interface.

    Every concrete strategy must implement apply(), which receives a
    RetryContext and returns the (mutated) RetryContext. Strategies must
    be stateless with respect to retrieval or generation state.
    """

    #: Machine-readable strategy identifier (matches HealingStrategy enum value).
    name: str = "BASE"

    @abstractmethod
    def apply(self, context: RetryContext) -> RetryContext:
        """
        Apply this strategy's recovery action to the retry context.

        Args:
            context: The current retry context. May be mutated in-place.

        Returns:
            The (possibly mutated) RetryContext.

        Raises:
            SafeFailureSignal: Only from SafeFailureStrategy, to signal
                an immediate safe refusal.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete Strategies
# ---------------------------------------------------------------------------

class IncreaseRetrievalStrategy(BaseHealingStrategy):
    """
    LOW_GROUNDEDNESS / INCOMPLETE_ANSWER — increase retrieval k.

    Provides the generator with a richer context window by fetching more
    documents, improving the probability that all supporting evidence is
    present when the answer is regenerated.
    """

    name = "LOW_GROUNDEDNESS"

    def apply(self, context: RetryContext) -> RetryContext:
        new_k = min(context.retrieval_k + _K_INCREMENT, _MAX_K)
        logger.debug(
            "IncreaseRetrievalStrategy: k %d → %d", context.retrieval_k, new_k
        )
        context.retrieval_k = new_k
        return context


class QueryRewriteStrategy(BaseHealingStrategy):
    """
    LOW_RELEVANCE — rewrite the user query using an LLM call.

    Produces a more targeted, semantically enriched query to improve
    the relevance of retrieved documents. Falls back to the original
    question if the LLM call fails.
    """

    name = "LOW_RELEVANCE"

    _REWRITE_PROMPT = (
        "You are a search query optimization assistant.\n"
        "Rewrite the following user question to be more specific and "
        "retrieval-friendly for a document search system.\n"
        "Output ONLY the rewritten question. No explanation, no prefix.\n\n"
        "Original question: {question}"
    )

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def apply(self, context: RetryContext) -> RetryContext:
        try:
            prompt = self._REWRITE_PROMPT.format(question=context.question)
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            rewritten = response.content.strip()
            if rewritten:
                logger.debug(
                    "QueryRewriteStrategy: '%s' → '%s'",
                    context.question,
                    rewritten,
                )
                context.rewritten_query = rewritten
            else:
                logger.warning(
                    "QueryRewriteStrategy: LLM returned empty rewrite. "
                    "Retaining original query."
                )
        except Exception as exc:
            logger.warning(
                "QueryRewriteStrategy: LLM rewrite failed (%s). "
                "Retaining original query.",
                exc,
            )
        return context


class StrictGroundingStrategy(BaseHealingStrategy):
    """
    HALLUCINATION — activate strict grounding mode.

    Sets use_strict_grounding=True on the RetryContext. The generator will
    receive a stricter system prompt that requires every factual claim to be
    directly traceable to verbatim text in the retrieved context.
    """

    name = "HALLUCINATION"

    def apply(self, context: RetryContext) -> RetryContext:
        logger.debug("StrictGroundingStrategy: enabling use_strict_grounding.")
        context.use_strict_grounding = True
        return context


class RemoveSuspiciousChunksStrategy(BaseHealingStrategy):
    """
    PROMPT_INJECTION — exclude previously retrieved chunks.

    Moves all chunk IDs from the most recent retrieval into the permanent
    exclusion list, then triggers a fresh retrieval from the remaining corpus.
    This removes potentially malicious chunks that may have carried injection
    payloads.
    """

    name = "PROMPT_INJECTION"

    def apply(self, context: RetryContext) -> RetryContext:
        new_exclusions = list(context.last_retrieved_chunk_ids)
        logger.debug(
            "RemoveSuspiciousChunksStrategy: excluding %d potentially "
            "malicious chunk(s). Total excluded: %d.",
            len(new_exclusions),
            len(context.excluded_chunk_ids) + len(new_exclusions),
        )
        context.excluded_chunk_ids.extend(new_exclusions)
        context.last_retrieved_chunk_ids = []
        return context


class RetryRetrievalStrategy(BaseHealingStrategy):
    """
    LOW_CONFIDENCE — reset and retry with original parameters.

    Clears any previously rewritten query so the orchestrator falls back to
    the original user question. This addresses transient retrieval or
    generation noise that caused low confidence.
    """

    name = "LOW_CONFIDENCE"

    def apply(self, context: RetryContext) -> RetryContext:
        logger.debug(
            "RetryRetrievalStrategy: clearing rewritten_query for fresh retry."
        )
        context.rewritten_query = None
        return context


class SafeFailureStrategy(BaseHealingStrategy):
    """
    EMPTY_CONTEXT — signal immediate safe failure.

    There is no meaningful recovery when the corpus contains no relevant
    documents. Raises SafeFailureSignal to short-circuit the retry loop so
    the controller can return a predefined safe refusal message.
    """

    name = "EMPTY_CONTEXT"

    def apply(self, context: RetryContext) -> RetryContext:
        logger.warning(
            "SafeFailureStrategy: empty context — signalling safe failure."
        )
        raise SafeFailureSignal(
            "No retrievable context found. Returning safe failure response."
        )
