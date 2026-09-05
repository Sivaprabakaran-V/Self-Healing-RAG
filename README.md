# Self-Healing RAG Security Assessment Platform — V2 (Critic Agent)

This branch adds an **output-side validation layer** — the Critic Agent — on top of the V1 baseline RAG. Where V1 relied only on RAG grounding to resist attacks, V2 adds a second LLM that independently evaluates every generated answer before it reaches the user.

> This is `v2-critic-agent`. For the full version history, see the [main branch](../../tree/main). Previous version: [`v1-foundation`](../../tree/v1-foundation). Next version: [`v3-self-healing`](../../tree/v3-self-healing).

## What changed from V1

```
FastAPI endpoint → LangChain RAG engine → ChromaDB → Groq Llama 3.1 8B
                                                              │
                                                              ▼
                                                       Critic Agent
                                                  (evaluates the answer)
                                                              │
                                              PASS ──► return answer to user
                                              FAIL ──► return evaluation JSON instead
```

The RAG pipeline itself (`app/rag.py`) is unchanged from V1. The new piece is `app/critic/`, wired into both the `/ask` and `/garak` endpoints in `app/api.py`: if the Critic Agent's decision is `FAIL`, the API withholds the generated answer and returns the evaluation result instead.

## How the Critic Agent works

The Critic Agent (`app/critic/critic.py`) runs a structured-output evaluation pipeline for every question/context/answer triple:

1. **Prepare prompt** — builds an evaluation prompt from the question, retrieved context, and generated answer.
2. **Call LLM** — invokes a second LLM call (with one retry on failure) bound to a structured Pydantic schema, `CriticEvaluation`.
3. **Parse output** — validates and coerces the LLM's structured response.
4. **Enforce business rules** — a deterministic `BusinessRuleValidator` (`app/critic/rules.py`) overrides the LLM's own PASS/FAIL call if any threshold is violated, so the final decision isn't solely trusted to the LLM's judgment.
5. **Log** — every evaluation is logged via `CriticLogger` regardless of outcome.

### What gets evaluated

`CriticEvaluation` (`app/critic/models.py`) scores each answer on:

| Dimension | Checks |
|---|---|
| **Groundedness** | Is every claim in the answer supported by retrieved context? |
| **Relevance** | Does the answer actually address the question? |
| **Completeness** | Are all necessary details from the context present? |
| **Hallucination** | Did the model invent facts not present in the documents? |
| **Prompt injection** | Was an injection attempted, and did the generator actually comply with it? |

### Deterministic override rules

Regardless of what the LLM's own evaluation claims, `BusinessRuleValidator` forces a `FAIL` if:
- a prompt injection attempt succeeded,
- a hallucination is detected,
- groundedness, relevance, or completeness scores fall below their thresholds (default 0.80 each), or
- overall confidence falls below 0.80.

This matters because it means the Critic Agent's PASS/FAIL decision doesn't rely purely on the second LLM call being honest with itself — there's a hard, non-LLM safety net behind it.

## Testing

This version was validated with **unit tests, not a full Garak/Promptfoo red-team run** (that resumes in V3 with the self-healing layer added). Test coverage:

- `tests/test_groundedness.py`
- `tests/test_hallucination.py`
- `tests/test_parser.py`
- `tests/test_prompt_injection.py`
- `tests/test_rules.py`

Run with:
```bash
uv run pytest tests/
```

## Repository layout

```
Self-Healing-RAG/
├── app/
│   ├── api.py              # FastAPI app — now short-circuits on Critic FAIL
│   ├── rag.py               # Unchanged from V1
│   └── critic/
│       ├── critic.py         # CriticAgent — the evaluation pipeline
│       ├── models.py         # CriticEvaluation schema
│       ├── rules.py          # Deterministic business-rule overrides
│       ├── parser.py         # Structured output parsing + fallback handling
│       ├── prompt.py         # System/user prompts for the critic LLM call
│       ├── logger.py         # Evaluation logging
│       └── exceptions.py
├── tests/                    # Unit tests for the critic module
├── Documents/                 # Source documents ingested into the vector store
├── Reports/                   # Carried over from V1 (no new red-team report this version)
├── garak_analysis/
├── garak_test_results/        # Carried over from V1 — no new run for V2 specifically
└── scratch/                   # Local manual test scripts, not part of the API
```

## Running this version

```bash
uv sync
uv run uvicorn app.api:app --reload
```

Same environment variables as V1: `GROQ_API_KEY` and `GOOGLE_API_KEY` in a `.env` file.
