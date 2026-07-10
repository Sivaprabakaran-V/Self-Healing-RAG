"""
Healing Controller — Data Models (V3)

Defines the Pydantic models and enumerations shared across all healing
sub-modules. No business logic lives here.
"""
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealingStrategy(str, Enum):
    """Enumeration of all available healing strategy identifiers."""

    LOW_GROUNDEDNESS = "LOW_GROUNDEDNESS"
    LOW_RELEVANCE    = "LOW_RELEVANCE"
    HALLUCINATION    = "HALLUCINATION"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    LOW_CONFIDENCE   = "LOW_CONFIDENCE"
    EMPTY_CONTEXT    = "EMPTY_CONTEXT"
    NO_ACTION        = "NO_ACTION"


class RetryContext(BaseModel):
    """
    Mutable context object passed through the retry orchestration loop.

    Each healing strategy mutates specific fields of this object. The
    RetryOrchestrator reads these fields to drive each retrieval and
    generation cycle. No RAG logic lives here.
    """

    question: str = Field(
        ...,
        description="The original user question (never mutated).",
    )
    retrieval_k: int = Field(
        default=5,
        description="Number of top documents to retrieve (k for MMR search).",
    )
    fetch_k: int = Field(
        default=20,
        description="Candidate pool size for MMR retrieval.",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description=(
            "LLM-rewritten query used in place of the original question "
            "during retrieval (LOW_RELEVANCE strategy)."
        ),
    )
    excluded_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk IDs permanently excluded from retrieval for this session "
            "(PROMPT_INJECTION strategy)."
        ),
    )
    last_retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk IDs from the most recent retrieval pass. Updated by the "
            "orchestrator after each retrieval so strategies can reference them."
        ),
    )
    use_strict_grounding: bool = Field(
        default=False,
        description=(
            "When True, the generator uses a stricter system prompt that "
            "requires verbatim citation (HALLUCINATION strategy)."
        ),
    )


class HealingAttempt(BaseModel):
    """Immutable record of a single healing retry attempt."""

    attempt_number: int = Field(
        ...,
        description="1-indexed retry attempt number.",
    )
    strategy: HealingStrategy = Field(
        ...,
        description="The healing strategy applied during this attempt.",
    )
    failure_reason: str = Field(
        ...,
        description="The Critic failure_reason that triggered this attempt.",
    )
    retrieval_k: int = Field(
        ...,
        description="The k value used for retrieval during this attempt.",
    )
    outcome: Literal["PASS", "FAIL", "EXHAUSTED", "SAFE_FAILURE"] = Field(
        ...,
        description="Outcome of this attempt as determined by the Critic.",
    )
    execution_time_s: float = Field(
        ...,
        description="Wall-clock seconds for this attempt (strategy + retrieval + generation + critic).",
    )


class HealingResult(BaseModel):
    """
    Comprehensive result returned by the HealingController after all
    retry attempts have been resolved (PASS, HEALED, FAIL, or SAFE_FAILURE).
    """

    healing_attempted: bool = Field(
        ...,
        description="True if at least one healing attempt was made.",
    )
    final_strategy: Optional[HealingStrategy] = Field(
        default=None,
        description="The healing strategy that produced the final answer.",
    )
    retry_count: int = Field(
        default=0,
        description="Total number of healing retries executed.",
    )
    attempts: list[HealingAttempt] = Field(
        default_factory=list,
        description="Ordered list of all healing attempt records.",
    )
    final_answer: str = Field(
        ...,
        description="The final answer text delivered to the user.",
    )
    status: Literal["PASS", "HEALED", "FAIL"] = Field(
        ...,
        description=(
            "PASS = no healing needed; "
            "HEALED = healing resolved the failure; "
            "FAIL = healing exhausted without resolution."
        ),
    )
    final_critic_evaluation: dict = Field(
        ...,
        description="The last CriticAgent evaluation result dict.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source files contributing to the final answer.",
    )
    retrieved_chunks: int = Field(
        default=0,
        description="Number of filtered context chunks used in the final answer.",
    )
    total_healing_time_s: float = Field(
        default=0.0,
        description="Total wall-clock seconds spent inside the HealingController.",
    )
    # ── Issue 5: extended metadata ──────────────────────────────────────────
    initial_failure_reason: str = Field(
        default="",
        description="The failure_reason from the first Critic FAIL that triggered healing.",
    )
    healing_success: bool = Field(
        default=False,
        description="True when status == 'HEALED' (healing resolved the Critic failure).",
    )
    healing_strategy: Optional[HealingStrategy] = Field(
        default=None,
        description="The healing strategy that produced the final answer.",
    )
    final_decision: str = Field(
        default="FAIL",
        description="The final decision: PASS or FAIL.",
    )
