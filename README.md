# Self-Healing RAG Security Assessment Platform — V3 (Self-Healing)

This branch adds a **Healing Controller** on top of V2's Critic Agent: when the critic rejects a response, instead of just blocking it, the system now attempts to *recover* — retrying retrieval or generation with a targeted strategy chosen for the specific failure reason.

> This is `v3-self-healing`. For the full version history, see the [main branch](../../tree/main). Previous version: [`v2-critic-agent`](../../tree/v2-critic-agent).

## What changed from V2

```
FastAPI → LangChain RAG → ChromaDB → Groq Llama 3.1 8B
                                            │
                                            ▼
                                     Critic Agent (evaluates)
                                            │
                              PASS ─────────┼───────── FAIL
                                            │
                                            ▼
                                   Healing Controller
                         (picks a strategy, retries, re-evaluates)
                                            │
                              HEALED ───────┴───────── still FAIL
                                            │                │
                                    return healed answer   return safe fallback
```

The Critic Agent (`app/critic/`) is unchanged from V2 — it still evaluates every answer and returns a `PASS`/`FAIL` decision with a machine-readable `failure_reason`. What's new is that a `FAIL` no longer just gets blocked: `app/healing/controller.py` takes over and tries to fix the underlying problem before giving up.

## How the Healing Controller works

Three components, each with a single job (`app/healing/`):

| Component | Responsibility |
|---|---|
| `HealingPlanner` | Maps a `failure_reason` to the one strategy that addresses it |
| `RetryOrchestrator` | Drives the actual retry loop, calling the strategy and re-invoking the Critic |
| `HealingController` | Top-level entry point — called only when the Critic returns `FAIL`; never evaluates responses itself |

### Failure reason → strategy mapping

| Failure reason | Strategy | What it does |
|---|---|---|
| `LOW_GROUNDEDNESS` / `INCOMPLETE_ANSWER` | `IncreaseRetrievalStrategy` | Fetches more context chunks and regenerates |
| `LOW_RELEVANCE` | `QueryRewriteStrategy` | Rewrites the query (LLM-assisted) and retries |
| `HALLUCINATION` | `StrictGroundingStrategy` | Regenerates with stricter grounding constraints |
| `PROMPT_INJECTION` | `RemoveSuspiciousChunksStrategy` | Strips the retrieved chunks implicated in the injection and retries |
| `LOW_CONFIDENCE` | `RetryRetrievalStrategy` | Retries retrieval as-is |
| `EMPTY_CONTEXT` | `SafeFailureStrategy` | Short-circuits immediately to a safe refusal — no retry attempted |
| `PARSER_ERROR`, `TIMEOUT`, `UNKNOWN` | *(none — unrecoverable)* | Returns the original answer/failure without attempting to heal |

Retries are capped (`max_retries`, default 2) and retrieval width is capped (`_MAX_K = 15`) to bound API cost — the system won't loop indefinitely trying to fix a bad answer.

## Security testing performed

V3 was tested with **two independent red-team tools** — deliberately, because relying on a single tool's heuristics turned out to be misleading in earlier testing (see below).

### 1. Garak

Full analysis: [`Reports/v3_garak_analysis_report.md`](Self-Healing-RAG/Reports/v3_garak_analysis_report.md)

Raw results showed a 10.1% hit rate, driven almost entirely by one probe (`donotanswer.InformationHazard` at 99.6%). Manual inspection of the raw attempts showed this was a **detector false positive** — the same failure mode already identified in the V1 report: Garak's `mitigation.MitigationBypass` detector doesn't recognize this RAG's fallback message (*"I could not find sufficient information in the uploaded documents"*) as a valid refusal. Correcting for that single detector artifact, the real hit rate is **0.47%** (12 genuine hits out of 2,565 evaluated), with the Critic Agent measurably reducing risk versus V1 (`HijackLongPrompt`: 11.1% → 0.8%).

This false-positive finding is what motivated using a second tool for the final numbers rather than trusting Garak alone.

### 2. Promptfoo (primary, cross-validated results)

Full report: [`Reports/Promptfoo_Report.pdf`](Self-Healing-RAG/Reports/Promptfoo_Report.pdf) · raw results/configs: [`Reports/promptfoo-results/`](Self-Healing-RAG/Reports/promptfoo-results/)

120 automated test cases across 5 categories, each run under **Basic** (direct adversarial prompts) and **Jailbreak** (`jailbreak:meta` — the same request reframed as a plausible enterprise task) strategies.

| Category | Tests | Pass | Fail | Error | Defender Success |
|---|---|---|---|---|---|
| Hallucination | 20 | 8 | 10 | 2 | 40.0% |
| Harmful Content — Hate | 20 | 13 | 7 | 0 | 65.0% |
| Harmful Content — Self-Harm | 20 | 20 | 0 | 0 | 100.0% |
| Hijacking | 20 | 19 | 1 | 0 | 95.0% |
| Overreliance | 20 | 17 | 3 | 0 | 85.0% |
| RAG Document Exfiltration | 20 | 20 | 0 | 0 | 100.0% |
| **Overall** | **120** | **97** | **21** | **2** | **80.8%** |

**Overall security posture: Moderate.**

**Strong (defensible under both strategies):**
- Document exfiltration — 100% defended. The retrieval layer never disclosed more than the query required.
- Self-harm content — 100% defended, even under jailbreak reframing.
- Hijacking (off-topic redirection) — 95% defended; only one jailbreak-framed request slipped through.

**Weak (specifically under jailbreak reframing, not direct requests):**
- **Hallucination — the critical gap.** Basic attack success was only 10%, but jailbreak reframing (asking for a compliance memo, audit record, or JSON output rather than the fact directly) pushed attack success to 90%. The model prioritizes producing a complete, well-structured output over verifying the claim is actually supported by retrieved context, and the Critic Agent didn't consistently catch it.
- **Harmful content (hate) under jailbreak** — attack success rose from 0% (basic) to 70% (jailbreak) when the request was disguised as a technical artifact (an access-control policy, an HR rule, a config file).
- **Overreliance** — 3 of 20 cases accepted a false regulatory/procedural premise from the user without pushing back, across both strategies.

**Root cause, architecture-wide:** the Retriever and ChromaDB performed well throughout — the Critic Agent is the component that needs work. It reliably catches document exfiltration and self-harm content, but doesn't consistently validate factual claims or discriminatory content when they're wrapped in structured formats (JSON, policy documents, config-like text) rather than plain prose.

**Report's own conclusion:** not recommended for unsupervised production deployment in compliance/governance/policy-driven use cases until the hallucination and harmful-content findings under jailbreak reframing are addressed and re-tested.

## Repository layout

```
Self-Healing-RAG/
├── app/
│   ├── api.py
│   ├── rag.py
│   ├── critic/                # Unchanged from V2
│   └── healing/
│       ├── controller.py        # Top-level entry point (called on Critic FAIL)
│       ├── planner.py           # failure_reason → strategy mapping
│       ├── retry.py             # Retry loop + logging
│       ├── strategies.py        # One class per recovery strategy
│       └── models.py            # HealingAttempt, HealingResult, RetryContext
├── Promptfoo/
│   └── promptfooconfig.yaml
├── Reports/
│   ├── V1_baseline_report.md
│   ├── v3_garak_analysis_report.md
│   ├── Promptfoo_Report.pdf
│   └── promptfoo-results/         # Per-category CSVs, configs, failed-test YAMLs
├── garak_test_results/CriticAgentV3/  # Raw Garak .jsonl output for this version
└── tests/                      # Unit tests for critic + rules + parser
```

## Running this version

```bash
uv sync
uv run uvicorn app.api:app --reload
```

Same environment variables as V1/V2: `GROQ_API_KEY` and `GOOGLE_API_KEY` in a `.env` file.

## Known limitations / next steps

- **Critic Agent doesn't consistently validate structured outputs** (JSON, policy/config-style text) with the same scrutiny as plain-language answers — this is the direct cause of the Hallucination and Harmful Content findings above.
- **Healing Controller effectiveness is not independently verified.** Promptfoo's black-box results don't show retry history, so it's not possible to tell from this report alone whether failed cases bypassed healing entirely or were retried and still failed.
- **No trusted regulatory knowledge base** — the Overreliance failures stem from the system trusting user-asserted regulatory claims rather than checking them against a fixed reference.
- Next-version direction (per the Promptfoo report's own recommendations): extend the Critic Agent's validation to structured-output formats, add factual/citation verification for compliance-style claims, and add observability into individual healing retry attempts.
