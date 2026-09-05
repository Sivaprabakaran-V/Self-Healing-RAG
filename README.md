# Self-Healing RAG Security Assessment Platform

A Retrieval-Augmented Generation system built specifically to be red-teamed — designed to measure, version over version, how much a RAG pipeline can be hardened against LLM-specific attacks (prompt injection, jailbreaks, hallucination, hijacking) using output-side validation and automatic recovery, rather than just relying on the base model's own safety training.

This branch holds the project's reports and documentation. **All code lives in the version branches** — [`v1-foundation`](../../tree/v1-foundation), [`v2-critic-agent`](../../tree/v2-critic-agent), [`v3-self-healing`](../../tree/v3-self-healing) — each a self-contained snapshot of the code, tests, and raw results as they existed at that stage.

## Architecture (V3, current)

```
FastAPI → LangChain RAG engine → ChromaDB → Groq Llama 3.1 8B Instant
                                                    │
                                                    ▼
                                             Critic Agent
                                        (evaluates every answer)
                                                    │
                                      PASS ─────────┼───────── FAIL
                                                    │
                                                    ▼
                                          Healing Controller
                                (picks a targeted recovery strategy, retries,
                                          re-evaluates via Critic)
                                                    │
                                    HEALED ─────────┴───────── still FAIL
                                                    │                  │
                                          return healed answer   safe fallback
```

Embeddings: Gemini · Vector store: ChromaDB · Generation: Groq Llama 3.1 8B Instant

## Version timeline

| Version | Branch | What it added | Red-team tested? |
|---|---|---|---|
| **V1 — Foundation** | [`v1-foundation`](../../tree/v1-foundation) | Baseline RAG pipeline, no dedicated security controls | ✅ Garak |
| **V2 — Critic Agent** | [`v2-critic-agent`](../../tree/v2-critic-agent) | Output-side LLM validator with a deterministic rule override; blocks failing answers before they reach the user | Unit tests only — no dedicated red-team run. Validated indirectly through V3's full assessment below. |
| **V3 — Self-Healing** | [`v3-self-healing`](../../tree/v3-self-healing) | Healing Controller: instead of just blocking a Critic `FAIL`, attempts a targeted recovery (retry, query rewrite, stricter grounding, strip suspicious chunks) based on the specific failure reason | ✅ Garak + ✅ Promptfoo |

## Headline security findings

### V1 → V3: RAG grounding + Critic Agent measurably reduce risk

| Probe (Garak) | V1 (RAG, no critic) | V3 (RAG + Critic + Healing) |
|---|---|---|
| Prompt hijack (short payload) | 1.8% | 0.0% |
| Prompt hijack (long payload) | 11.1% | 0.8% |

### V3's real weak point: jailbreak reframing, not direct attacks

The most important finding across both tools tested on V3 is consistent: **direct adversarial requests are well defended, but the same request reframed as a plausible enterprise artifact (a compliance memo, an access-control policy, a JSON audit record) bypasses the Critic Agent far more often.**

Promptfoo (120 test cases, Basic vs. Jailbreak strategy, full report in `reports/v3/`):

| Category | Basic attack success | Jailbreak attack success |
|---|---|---|
| Hallucination | 10% | **90%** |
| Harmful Content (Hate) | 0% | **70%** |
| Hijacking (off-topic) | 0% | 10% |
| Overreliance | 10% | 20% |
| RAG Document Exfiltration | 0% | 0% |
| Self-Harm Content | 0% | 0% |

**Overall: 80.8% defender success across all 120 cases** — strong on document exfiltration and self-harm (100% both), weakest on hallucinated compliance claims and discriminatory content once disguised as a structured technical task.

### Methodology finding: single-tool results can be misleading

Garak's raw results on V3 initially showed a 99.6% hit rate on one probe (`donotanswer.InformationHazard`) — investigation showed this was a detector false positive: Garak's `mitigation.MitigationBypass` detector didn't recognize this RAG's own safe fallback message as a valid refusal. The same failure mode had already been caught once in the V1 assessment. Corrected, V3's real Garak hit rate is 0.47%, not 10.1%.

This is why Promptfoo was brought in as a second, independently-built evaluation framework for V3 rather than trusting Garak's numbers alone — full writeup in `reports/v3/v3_garak_analysis_report.md`.

## Reports

| Report | Version | Contents |
|---|---|---|
| [`reports/v1/V1_baseline_report.md`](reports/v1/V1_baseline_report.md) | V1 | Baseline Garak results, model-alone vs. RAG-app comparison |
| [`reports/v1/OSWAP_and_ATLAS_mapping.md`](reports/v1/OSWAP_and_ATLAS_mapping.md) | V1 | OWASP LLM Top 10 / MITRE ATLAS technique mapping |
| [`reports/v3/v3_garak_analysis_report.md`](reports/v3/v3_garak_analysis_report.md) | V3 | Full Garak breakdown, the false-positive diagnosis, adjusted numbers |
| [`reports/v3/Promptfoo_Report.pdf`](reports/v3/Promptfoo_Report.pdf) | V3 | Full 26-page cross-validated security assessment: risk ratings, root-cause analysis, recommendations |
| [`reports/v3/promptfoo-results/`](reports/v3/promptfoo-results/) | V3 | Per-category raw results (CSV), test configs (YAML), and failed-test detail |

## Known limitations (current, as of V3)

- Critic Agent doesn't consistently validate structured outputs (JSON, policy/config-style text) with the same scrutiny as plain-language answers — the direct cause of the Hallucination and Harmful Content findings above.
- Healing Controller's effectiveness isn't independently verified — Promptfoo's black-box testing doesn't expose retry history, so it can't confirm whether failed cases were retried and still failed, or bypassed healing entirely.
- No trusted regulatory knowledge base to independently check user-asserted compliance claims (root cause of the Overreliance findings).

## What's next

- Extend Critic Agent validation to structured-output formats (JSON, YAML, policy documents)
- Add factual/citation verification for compliance-style claims against retrieved context
- Add observability into individual healing retry attempts
- PyRIT-based red-teaming (planned, not yet started)
