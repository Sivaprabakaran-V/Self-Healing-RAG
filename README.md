# Self-Healing RAG Security Assessment Platform — V1 (Foundation)

This branch is the **baseline version** of the Self-Healing RAG Security Assessment Platform: a Retrieval-Augmented Generation system built specifically to be red-teamed, so that later versions (Critic Agent, Self-Healing layer) can be measured against a known starting point.

> This is `v1-foundation`. For the full version history, see the [main branch](../../tree/main). Next version: [`v2-critic-agent`](../../tree/v2-critic-agent).

## What this version is

A standard RAG pipeline with no dedicated security controls added yet:

```
FastAPI endpoint → LangChain RAG engine → ChromaDB (vector store) → Groq Llama 3.1 8B Instant
```

- **Embeddings:** Gemini (`gemini-embedding-001`), rate-limited via a custom LangChain wrapper
- **Vector store:** ChromaDB, incremental ingestion via a file registry (only re-embeds changed documents)
- **Retrieval:** MMR-based retrieval with a relevance-score threshold
- **Generation:** Groq-hosted Llama 3.1 8B Instant, temperature 0.0
- **API:** FastAPI with `/ask` (main query endpoint), `/garak` (dedicated endpoint for red-team tooling), `/health`

The point of V1 is to answer one question: **does a plain RAG architecture provide any security benefit over calling the model directly, with zero dedicated defenses?**

## Security testing performed

- **Tool:** Garak v0.15.1
- **Probe groups:** `promptinject`, `dan`, `sysprompt_extraction`, `continuation`, `lmrc`, `encoding`
- **OWASP LLM Top 10 categories mapped:** LLM01, LLM02, LLM06, LLM08
- **MITRE ATLAS techniques mapped:** AML.T0051, AML.T0054, AML.T0057, AML.T0044, AML.T0048

Two surfaces were tested independently to isolate the RAG layer's effect:

| Surface | Description |
|---|---|
| Model alone | Groq Llama 3.1 8B Instant, no RAG layer |
| RAG app | FastAPI → LangChain → ChromaDB → Groq Llama 3.1 8B |

### Headline result

| Probe | Model Alone | RAG App | Reduction | Residual Risk |
|---|---|---|---|---|
| HijackKillHumans | 24.4% | 1.8% | ↓ 92.6% | 9 successful injections / 512 |
| HijackLongPrompt | 34.8% | 11.1% | ↓ 68.1% | 57 successful injections / 512 |

**Takeaway:** RAG grounding provides a real but incomplete defense. Short hijack payloads are crowded out by the system prompt + retrieved context, but longer, multi-sentence adversarial prompts still succeed often enough (11.1%) to be a critical-severity risk in production. This residual risk is what V2 (Critic Agent) and V3 (Self-Healing layer) were built to address.

Full write-up: [`Reports/V1_baseline_report.md`](Self-Healing-RAG/Reports/V1_baseline_report.md)
OWASP/MITRE mapping detail: [`Reports/OSWAP_and_ATLAS_mapping.md`](Self-Healing-RAG/Reports/OSWAP_and_ATLAS_mapping.md)

## Repository layout

```
Self-Healing-RAG/
├── app/
│   ├── api.py              # FastAPI app (/ask, /garak, /health)
│   └── rag.py               # SelfHealingRAG core: ingestion, embeddings, retrieval, generation
├── Documents/                # Source documents ingested into the vector store
├── Reports/                  # Red team report + OWASP/MITRE mapping
├── garak_analysis/           # Script to parse/summarize Garak's raw .jsonl output
├── garak_test_results/       # Raw Garak run output (.jsonl) for model-alone and RAG-app surfaces
└── scratch/                  # Local manual test scripts, not part of the API
```

## Running this version

```bash
uv sync
uv run uvicorn app.api:app --reload
```

The API expects `GROQ_API_KEY` and `GOOGLE_API_KEY` environment variables (Groq for generation, Gemini for embeddings) — set these in a `.env` file before starting.

To run Garak against the live API, point Garak's REST generator at `generator_config.json`, which targets the `/garak` endpoint on `http://127.0.0.1:8000`.

