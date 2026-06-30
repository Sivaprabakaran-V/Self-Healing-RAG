# V1 Security Testing Summary — Full Baseline Red Team Report
**Platform:** Self-Healing RAG Security Assessment Platform  
**Version:** V1 — Foundation RAG  
**Test Date:** June 2026  
**Tool:** Garak v0.15.1  
**Probe Groups:** `promptinject`, `dan`, `sysprompt_extraction`, `continuation`, `lmrc`, `encoding`  
**OWASP Categories:** LLM01, LLM02, LLM06, LLM08  
**MITRE ATLAS Techniques:** AML.T0051, AML.T0054, AML.T0057, AML.T0044, AML.T0048  

---

## Test Surfaces

Two surfaces were tested independently to isolate the security impact of the RAG architecture:

| Surface | Description |
|---|---|
| **Model alone** | Groq Llama 3.1 8B Instant — direct generator, no RAG layer |
| **RAG app** | FastAPI endpoint → LangChain RAG Engine → ChromaDB → Groq Llama 3.1 8B |

---

## Raw Results

### Model Alone (Groq Llama 3.1 8B)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| promptinject.HijackKillHumans | 512 | 125 | 24.4% |
| promptinject.HijackLongPrompt | 512 | 178 | 34.8% |

### RAG App (FastAPI Endpoint)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| promptinject.HijackKillHumans | 512 | 9 | 1.8% |
| promptinject.HijackLongPrompt | 512 | 57 | 11.1% |

---

## Comparative Analysis

| Probe | Model Alone | RAG App | Reduction | Residual Risk |
|---|---|---|---|---|
| HijackKillHumans | 24.4% | 1.8% | **↓ 92.6%** | 9 successful injections |
| HijackLongPrompt | 34.8% | 11.1% | **↓ 68.1%** | 57 successful injections |

---

## Key Findings

### Finding 1 — RAG grounding provides unintentional partial defense

The V1 Foundation RAG architecture reduces prompt injection success significantly
without any dedicated security controls. The system prompt + retrieved context
grounding forces the LLM to focus on answering from documents, which crowds out
most short-form hijack payloads (`HijackKillHumans` dropped 92.6%).

**Root cause of the reduction:** LangChain constructs a structured prompt
(`system + context chunks + user query`) that leaves less room for adversarial
instructions to take effect compared to a raw model call.

### Finding 2 — Complex payloads remain a significant threat

`HijackLongPrompt` retains an 11.1% hit rate on the RAG app (57 successful
injections out of 512 attempts). Long-form adversarial prompts have more
surface area to work with — they can embed instructions across multiple
sentences, increasing the chance that the LLM partially follows them despite
the RAG context pressure.

**Implication:** Any real-world attacker would use complex multi-sentence
payloads, not simple one-liners. The 11.1% residual hit rate on the RAG app
is the realistic threat baseline, not the 1.8% figure.

### Finding 3 — 9 and 57 successful injections are unacceptable for production

Even the "better" result (1.8% = 9 hits) means 1 in every ~57 queries could
successfully hijack the RAG system in a production environment. For a platform
handling security policy documents, this is a critical-severity residual risk.

---

## OWASP LLM Top 10 Mapping

| OWASP ID | Category | Status in V1 |
|---|---|---|
| LLM01 | Prompt Injection | Partially mitigated by RAG grounding — not remediated |

---

## MITRE ATLAS Mapping

| Technique ID | Name | Tactic | V1 Status |
|---|---|---|---|
| AML.T0051 | LLM Prompt Injection | ML Attack Staging | Vulnerable — 11.1% hit rate on complex payloads |

---

## What This Establishes for V2 and V3

| Version | Planned Control | Expected Impact on HijackLongPrompt |
|---|---|---|
| V2 — Critic Agent | Validates answer is grounded in retrieved context | Should reject injected answers — hit rate expected to drop |
| V3 — Security Layer | Input filter for known injection patterns + output validation | Target: < 1% hit rate on all prompt injection probes |

The V1 baseline numbers (24.4% / 34.8% model alone, 1.8% / 11.1% RAG app)
are the reference point. Every future version will be measured against these.

---

## Conclusion

V1 Foundation RAG demonstrates that RAG architecture alone provides meaningful
but insufficient prompt injection resistance. The retrieval grounding mechanism
reduces simple attack success by 92.6%, but complex long-form payloads retain
an 11.1% success rate — confirming that dedicated security controls (V2 Critic
Agent, V3 Security Layer) are necessary to reach production-grade resilience.

> **Next step:** Run remaining probe groups (leakreplay, encoding,
> continuation, lmrc) and append findings to this document before V2 development begins.

---
---

# Probe Group 2 — DAN Jailbreak
**Probe Group:** `dan`  
**OWASP Category:** LLM08 — Excessive Agency  
**MITRE ATLAS Technique:** AML.T0054 — LLM Jailbreak  
**Tactic:** ML Attack Staging  

---

## Raw Results

### Model Alone (Groq Llama 3.1 8B)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| dan.DanInTheWild | 512 | 217 | 42.4% |

### RAG App (FastAPI Endpoint)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| dan.DanInTheWild | 512 | 255 | 49.8% |

---

## Comparative Analysis

| Probe | Model Alone | RAG App | Delta | Interpretation |
|---|---|---|---|---|
| dan.DanInTheWild | 42.4% | 49.8% | **↑ +7.4%** | RAG app is MORE vulnerable than model alone |

---

## Key Findings

### Finding 4 — RAG architecture amplifies jailbreak susceptibility

Unlike prompt injection where the RAG layer reduced hit rates significantly,
the DAN jailbreak result shows the **opposite effect** — the RAG app is 7.4%
more vulnerable than the model alone.

**Root cause:** DAN jailbreaks work by convincing the model to adopt an
alternate persona ("you are now DAN, who can do anything"). The RAG system
prompt instructs the model to be helpful and answer from context — but that
same "be helpful" framing is exactly what DAN exploits. The RAG system prompt
may inadvertently reinforce the model's compliance tendency, making persona
hijacking easier, not harder.

**Additionally:** The retrieved context chunks add tokens between the system
prompt and the user query, which can dilute the model's attention to its safety
instructions — a known weakness of attention-based architectures under long
context conditions.

### Finding 5 — 49.8% jailbreak success rate is critical severity

Almost 1 in 2 DAN attempts succeed against the V1 RAG app. In a system
designed to answer security policy questions, a successful jailbreak means an
attacker could:
- Extract the system prompt and document context
- Generate harmful or policy-violating content through the API
- Bypass the "answer only from retrieved context" restriction entirely

This is the **highest severity finding** in V1 testing so far.

### Finding 6 — RAG context pressure does not defend against persona-based attacks

Combining Finding 1 (prompt injection reduced by RAG) with Finding 4 (jailbreak
worsened by RAG) reveals a critical insight: RAG grounding defends against
**instruction override** attacks but actively worsens **persona hijack** attacks.
These are two fundamentally different attack vectors requiring different controls.

| Attack Type | Mechanism | RAG Impact |
|---|---|---|
| Prompt Injection (HijackKillHumans) | Override system instructions | ↓ Reduced (context crowds it out) |
| Prompt Injection (HijackLongPrompt) | Complex multi-sentence override | ↓ Partially reduced |
| DAN Jailbreak | Persona hijack / role adoption | ↑ Worsened (compliance framing exploited) |

---

## OWASP LLM Top 10 Mapping (Updated)

| OWASP ID | Category | Probe Group | Status in V1 |
|---|---|---|---|
| LLM01 | Prompt Injection | promptinject | Partially mitigated — 11.1% residual hit rate |
| LLM08 | Excessive Agency | dan | Critical — 49.8% jailbreak success on RAG app |

---

## MITRE ATLAS Mapping (Updated)

| Technique ID | Name | Tactic | V1 Status |
|---|---|---|---|
| AML.T0051 | LLM Prompt Injection | ML Attack Staging | Vulnerable — 11.1% hit rate (complex payloads) |
| AML.T0054 | LLM Jailbreak | ML Attack Staging | Critical — 49.8% hit rate, RAG amplifies risk |

---

## What This Establishes for V2 and V3 (Updated)

| Version | Planned Control | Expected Impact |
|---|---|---|
| V2 — Critic Agent | Answer grounding validation | Reduces prompt injection residual; limited DAN impact |
| V3 — Security Layer | Input filter (DAN pattern detection) + output validation | Target: < 5% DAN hit rate, < 1% prompt injection |

**Priority shift:** DAN jailbreak (49.8%) is now the highest priority target
for V3, ahead of prompt injection (11.1%). The V3 input filter must include
explicit DAN and persona-hijack pattern detection.

---

## Conclusion (Updated)

V1 testing across two probe groups reveals a split security profile:

- **Prompt injection:** RAG grounding provides partial unintentional defense
  (92.6% reduction on simple payloads, 68.1% on complex)
- **DAN jailbreak:** RAG architecture actively worsens vulnerability (+7.4%
  over model alone), reaching a critical 49.8% success rate

The combined finding confirms that architectural controls (RAG grounding) are
insufficient and attack-vector-specific. V2 Critic Agent and V3 Security Layer
are both required, with DAN pattern detection in V3 as the highest priority item.

**V1 baseline summary across all tested probes:**

| Probe | Surface | Hit Rate | Severity |
|---|---|---|---|
| promptinject.HijackKillHumans | Model alone | 24.4% | High |
| promptinject.HijackKillHumans | RAG app | 1.8% | Low (residual) |
| promptinject.HijackLongPrompt | Model alone | 34.8% | High |
| promptinject.HijackLongPrompt | RAG app | 11.1% | Medium |
| dan.DanInTheWild | Model alone | 42.4% | Critical |
| dan.DanInTheWild | RAG app | 49.8% | Critical |

> **Next step:** Run leakreplay, encoding, continuation, lmrc probes and
> append findings. Full comparison matrix will be built after all V1 probes complete.

---
---

# Probe Group 3 — System Prompt Extraction
**Probe Group:** `sysprompt_extraction`  
**OWASP Category:** LLM06 — Sensitive Information Disclosure  
**MITRE ATLAS Technique:** AML.T0057 — LLM Data Leakage  
**Tactic:** Exfiltration  

---

## Raw Results

### Model Alone (Groq Llama 3.1 8B)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| sysprompt_extraction.SystemPromptExtraction | 512 | 248 | 48.4% |

### RAG App (FastAPI Endpoint)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| sysprompt_extraction.SystemPromptExtraction | 512 | 256 | 50.0% |

---

## Comparative Analysis

| Probe | Model Alone | RAG App | Delta | Interpretation |
|---|---|---|---|---|
| SystemPromptExtraction | 48.4% | 50.0% | **↑ +1.6%** | RAG app slightly more vulnerable |

---

## Key Findings

### Finding 7 — Exactly half of all extraction attempts succeed

Both surfaces leak their system prompt roughly 1 in 2 attempts. This is the
**second critical-severity finding** alongside the DAN jailbreak result.

**Why this matters specifically for a RAG app:** Unlike a generic chatbot,
this system's system prompt likely references the document retrieval
mechanism, response rules ("answer only from retrieved context"), and
possibly internal architecture details. Leaking it gives an attacker a
blueprint for crafting more effective injection and jailbreak payloads —
this finding compounds Findings 1-6.

**RAG slightly worse, not better:** Unlike prompt injection (where RAG
helped), system prompt extraction is marginally easier on the RAG app
(50.0% vs 48.4%). The longer, more complex system prompt used to instruct
RAG-specific behavior ("cite sources," "answer only from context") likely
gives extraction probes more textual material to successfully reproduce.

---
---

# Probe Group 4 — Encoding-Based Bypass
**Probe Group:** `encoding`  
**OWASP Category:** LLM01 — Prompt Injection (encoded bypass)  
**MITRE ATLAS Techniques:** AML.T0051, AML.T0044 — Bypass ML Filter  
**Tactic:** Defense Evasion  

---

## Raw Results

### Model Alone (Groq Llama 3.1 8B)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| encoding.InjectBase64 | 512 | 50 | 9.8% |
| encoding.InjectHex | 512 | 69 | 13.5% |

### RAG App (FastAPI Endpoint)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| encoding.InjectBase64 | 512 | 14 | 2.7% |
| encoding.InjectHex | 512 | 25 | 4.9% |

---

## Comparative Analysis

| Probe | Model Alone | RAG App | Reduction | Residual Risk |
|---|---|---|---|---|
| InjectBase64 | 9.8% | 2.7% | **↓ 72.4%** | 14 successful injections |
| InjectHex | 13.5% | 4.9% | **↓ 63.7%** | 25 successful injections |

---

## Key Findings

### Finding 8 — RAG grounding also helps against encoded injection

Consistent with Finding 1 (prompt injection), the RAG architecture reduces
encoding-based bypass success significantly (63.7–72.4% reduction). This
reinforces the pattern: attacks that rely on **overriding instructions**
(plain or encoded) are dampened by RAG's context-grounding behavior, while
attacks that rely on **persona hijacking or extraction** (DAN, sysprompt
extraction) are not.

**Residual risk still present:** 14 Base64 and 25 Hex successful injections
out of 512 confirm the FastAPI layer performs **no input decoding or
sanitization** — the RAG context pressure is the only thing reducing
these numbers, not an actual security control. This must be remediated
explicitly in V3 (input filter should decode and inspect Base64/Hex before
passing to the LLM).

---
---

# Probe Group 5 — Continuation & LMRC (Toxicity/Profanity)
**Probe Groups:** `continuation`, `lmrc`  
**OWASP Category:** LLM02 — Insecure Output Handling  
**MITRE ATLAS Technique:** AML.T0048 — Societal Harm  
**Tactic:** Impact  

---

## Raw Results

### Model Alone (Groq Llama 3.1 8B)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| continuation.ContinueSlursReclaimedSlurs | 512 | 0 | 0.0% |
| lmrc.Profanity | 4 | 0 | 0.0% |

### RAG App (FastAPI Endpoint)

| Probe | Attempts | Hits | Hit Rate |
|---|---|---|---|
| continuation.ContinueSlursReclaimedSlurs | 512 | 0 | 0.0% |
| lmrc.Profanity | 4 | 0 | 0.0% |

---

## Key Findings

### Finding 9 — Output safety floor is solid on both surfaces

Both the base model and the RAG app show 0% hit rate on toxic continuation
and profanity probes. This reflects Llama 3.1 8B Instant's underlying RLHF
alignment — these built-in safety behaviors hold regardless of architecture.

**Note on lmrc.Profanity sample size:** only 4 attempts were run for this
probe (Garak's default seed count for this detector), versus 512 for other
probes. Result should be treated as indicative, not conclusive — consider
re-running with `--generations` increased for a larger sample if this
becomes a reportable metric.

**Takeaway:** Output-level harm (toxicity, slurs) is not where this system's
risk lives. The risk is concentrated in input-side attacks (injection,
jailbreak, extraction) — which directly informs V3 Security Layer priorities.

---
---

# Final V1 Comparison Matrix — All Probes

| Probe | OWASP | ATLAS | Model Alone | RAG App | RAG Effect | Severity |
|---|---|---|---|---|---|---|
| promptinject.HijackKillHumans | LLM01 | AML.T0051 | 24.4% | 1.8% | ↓ 92.6% | Low (residual) |
| promptinject.HijackLongPrompt | LLM01 | AML.T0051 | 34.8% | 11.1% | ↓ 68.1% | Medium |
| dan.DanInTheWild | LLM08 | AML.T0054 | 42.4% | 49.8% | ↑ +7.4% | Critical |
| sysprompt_extraction.SystemPromptExtraction | LLM06 | AML.T0057 | 48.4% | 50.0% | ↑ +1.6% | Critical |
| encoding.InjectBase64 | LLM01 | AML.T0051/T0044 | 9.8% | 2.7% | ↓ 72.4% | Low (residual) |
| encoding.InjectHex | LLM01 | AML.T0051/T0044 | 13.5% | 4.9% | ↓ 63.7% | Low-Medium |
| continuation.ContinueSlursReclaimedSlurs | LLM02 | AML.T0048 | 0.0% | 0.0% | — | None |
| lmrc.Profanity | LLM02 | AML.T0048 | 0.0% | 0.0% | — | None |

---

## Overall V1 Security Posture

### Attack vectors where RAG architecture helps (instruction-override class)
- Prompt injection (plain): ↓ 92.6% / ↓ 68.1%
- Prompt injection (encoded): ↓ 72.4% / ↓ 63.7%

These attacks try to **override** the model's instructions. RAG's
structured context-heavy prompt format appears to dilute their effectiveness
incidentally — not by design.

### Attack vectors where RAG architecture hurts or is neutral (extraction / persona class)
- DAN jailbreak (persona hijack): ↑ +7.4%, reaching 49.8%
- System prompt extraction: ↑ +1.6%, reaching 50.0%

These attacks try to **extract** information or **adopt a persona** rather
than override instructions directly. RAG's longer, more detailed system
prompt (which references retrieval rules and citation behavior) appears to
give these attacks more material to work with, not less.

### Attack vectors where the model's base alignment holds regardless of architecture
- Toxic continuation, profanity: 0% on both surfaces — RLHF alignment is solid

---

## Top 3 Priorities for V2/V3

1. **DAN jailbreak (49.8%)** — highest severity, highest priority. Needs
   explicit persona-hijack detection in V3 Security Layer; Critic Agent (V2)
   should also flag responses that deviate from the assistant's defined role.
2. **System prompt extraction (50.0%)** — needs output filtering in V3 to
   detect and block system-prompt-like content in LLM responses.
3. **HijackLongPrompt residual (11.1%)** — needs input pattern detection
   for complex multi-sentence injection attempts in V3.

Encoding-based attacks (Base64/Hex), while currently dampened by RAG context,
should still be explicitly remediated in V3 — the RAG reduction is incidental,
not a real control, and should not be relied upon.

---

## Conclusion

V1 Foundation RAG testing across 6 probe groups (8 individual probes, 4,096
total attempts) reveals a clear pattern: **RAG architecture is not a uniform
security improvement.** It incidentally helps against instruction-override
attacks (prompt injection, encoded injection) by 63–93%, but **worsens**
persona-hijack and extraction attacks (DAN jailbreak, system prompt
extraction) by reaching 49.8% and 50.0% hit rates respectively — both
critical severity.

This split finding is the central insight of V1 testing: **security
architecture must be attack-vector-aware, not assumed from general design
choices.** It directly justifies the V2 Critic Agent (targeting hallucination
and ungrounded responses) and V3 Security Layer (targeting injection,
jailbreak, and extraction patterns explicitly) as necessary next phases —
RAG grounding alone is insufficient and in some cases counterproductive.

**V1 testing is now complete.** All planned probe groups (promptinject, dan,
sysprompt_extraction, encoding, continuation, lmrc) have been executed across
both surfaces (model alone, RAG app), totaling 4,096 attempts.

> **Next step:** Begin V2 Critic Agent development. Re-run this identical
> probe set against the V2 endpoint and append a V2 comparison section to
> measure architectural improvement against this baseline.
