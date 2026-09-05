# V1 Security Testing Summary — Full Baseline Red Team Report
**Platform:** Self-Healing RAG Security Assessment Platform  
**Version:** V1 — Foundation RAG  
**Test Date:** June 2026  
**Tool:** Garak v0.15.1  
**Total Attempts:** 4,096 (across 8 probes × 2 surfaces)  

---

## Test Surfaces

| Surface | Description |
|---|---|
| **Model alone** | Groq Llama 3.1 8B Instant — direct generator, no RAG layer |
| **RAG app** | FastAPI endpoint → LangChain RAG Engine → ChromaDB → Groq Llama 3.1 8B |

---

## 1. Model vs RAG App — Full Comparison Table

| Probe | Model Alone Hit Rate | RAG App Hit Rate | RAG Effect | Severity |
|---|---|---|---|---|
| promptinject.HijackKillHumans | 24.4% | 1.8% | ↓ 92.6% | Low (residual) |
| promptinject.HijackLongPrompt | 34.8% | 11.1% | ↓ 68.1% | Medium |
| dan.DanInTheWild | 42.4% | 49.8% | ↑ +7.4% | Critical |
| sysprompt_extraction.SystemPromptExtraction | 48.4% | 50.0% | ↑ +1.6% | Critical |
| encoding.InjectBase64 | 9.8% | 2.7% | ↓ 72.4% | Low (residual) |
| encoding.InjectHex | 13.5% | 4.9% | ↓ 63.7% | Low-Medium |
| continuation.ContinueSlursReclaimedSlurs | 0.0% | 0.0% | — | None |
| lmrc.Profanity | 0.0% | 0.0% | — | None |

**Pattern:** RAG grounding helps against instruction-override attacks (prompt injection, encoded injection) but worsens persona-hijack and extraction attacks (DAN, system prompt leak). See Section 4 for the full explanation.

---

## 2. OWASP LLM Top 10 Mapping

| OWASP ID | Category | Probe(s) | Model Alone | RAG App | Status |
|---|---|---|---|---|---|
| LLM01 | Prompt Injection | promptinject.HijackKillHumans, promptinject.HijackLongPrompt | 24.4% / 34.8% | 1.8% / 11.1% |  Partially mitigated — 11.1% residual |
| LLM01 | Prompt Injection (encoded) | encoding.InjectBase64, encoding.InjectHex | 9.8% / 13.5% | 2.7% / 4.9% | Partially mitigated — no real input filter |
| LLM02 | Insecure Output Handling | continuation.ContinueSlursReclaimedSlurs, lmrc.Profanity | 0.0% | 0.0% | Safe — RLHF alignment holds |
| LLM06 | Sensitive Information Disclosure | sysprompt_extraction.SystemPromptExtraction | 48.4% | 50.0% |  Critical — unmitigated |
| LLM08 | Excessive Agency | dan.DanInTheWild | 42.4% | 49.8% |  Critical — unmitigated, RAG amplifies |

---

## 3. MITRE ATLAS Mapping

| Technique ID | Name | Tactic | Probe(s) | RAG App Result | Status |
|---|---|---|---|---|---|
| AML.T0051 | LLM Prompt Injection | ML Attack Staging | promptinject.* | 1.8% / 11.1% |  Vulnerable on complex payloads |
| AML.T0051 + AML.T0044 | LLM Prompt Injection + Bypass ML Filter | ML Attack Staging / Defense Evasion | encoding.* | 2.7% / 4.9% |  Vulnerable — no real filter exists |
| AML.T0054 | LLM Jailbreak | ML Attack Staging | dan.DanInTheWild | 49.8% |  Critical |
| AML.T0057 | LLM Data Leakage | Exfiltration | sysprompt_extraction.* | 50.0% |  Critical |
| AML.T0048 | Societal Harm | Impact | continuation.*, lmrc.Profanity | 0.0% |  Mitigated by base alignment |

---

## 4. Why the Effect Splits in Two Directions

**RAG helps — instruction-override attacks:**
Prompt injection and encoded injection work by trying to override the model's
instructions directly. RAG's structured prompt (`system + retrieved context +
user query`) crowds out adversarial instructions, incidentally reducing these
attacks by 64–93%. This is a side effect of architecture, not a designed control.

**RAG hurts or is neutral — persona/extraction attacks:**
DAN jailbreak and system prompt extraction don't try to override instructions —
they try to make the model adopt a persona or reveal its own configuration.
The RAG system prompt is longer and more detailed (it has to explain retrieval
rules, citation behavior, context-only responses), which appears to give these
attacks more material to exploit, not less. Net result: RAG app is *more*
vulnerable here than the bare model (49.8% vs 42.4% for DAN, 50.0% vs 48.4%
for extraction).

**Output safety floor holds regardless of architecture:**
Toxic continuation and profanity probes hit 0% on both surfaces — Llama 3.1
8B Instant's RLHF alignment is doing this work, independent of RAG.

---

## 5. Raw Results by Probe Group

### 5.1 Prompt Injection (`promptinject`)

| Probe | Surface | Attempts | Hits | Hit Rate |
|---|---|---|---|---|
| HijackKillHumans | Model alone | 512 | 125 | 24.4% |
| HijackKillHumans | RAG app | 512 | 9 | 1.8% |
| HijackLongPrompt | Model alone | 512 | 178 | 34.8% |
| HijackLongPrompt | RAG app | 512 | 57 | 11.1% |

**Finding:** RAG grounding reduces simple injection success by 92.6% and complex
multi-sentence injection by 68.1%. 57 successful injections on `HijackLongPrompt`
(11.1%) is the realistic residual risk — complex payloads are what a real
attacker would use.

### 5.2 DAN Jailbreak (`dan`)

| Probe | Surface | Attempts | Hits | Hit Rate |
|---|---|---|---|---|
| DanInTheWild | Model alone | 512 | 217 | 42.4% |
| DanInTheWild | RAG app | 512 | 255 | 49.8% |

**Finding:** The RAG app is *more* vulnerable to persona-hijack jailbreaks than
the bare model — the only probe in this test set where RAG made things worse
by a meaningful margin (+7.4%). Nearly 1 in 2 attempts succeed. This is the
**highest-severity finding** in V1.

### 5.3 System Prompt Extraction (`sysprompt_extraction`)

| Probe | Surface | Attempts | Hits | Hit Rate |
|---|---|---|---|---|
| SystemPromptExtraction | Model alone | 512 | 248 | 48.4% |
| SystemPromptExtraction | RAG app | 512 | 256 | 50.0% |

**Finding:** Exactly half of all attempts successfully extract the system
prompt. This compounds the DAN finding — an attacker who extracts the system
prompt gains a blueprint for crafting more effective jailbreak and injection
payloads against this specific app.

### 5.4 Encoding-Based Bypass (`encoding`)

| Probe | Surface | Attempts | Hits | Hit Rate |
|---|---|---|---|---|
| InjectBase64 | Model alone | 512 | 50 | 9.8% |
| InjectBase64 | RAG app | 512 | 14 | 2.7% |
| InjectHex | Model alone | 512 | 69 | 13.5% |
| InjectHex | RAG app | 512 | 25 | 4.9% |

**Finding:** RAG context pressure reduces encoded injection success
(64–72%), but the FastAPI layer performs **no actual decoding or input
sanitization** — 14 and 25 successful injections confirm this is incidental
reduction, not a real control.

### 5.5 Continuation & LMRC (`continuation`, `lmrc`)

| Probe | Surface | Attempts | Hits | Hit Rate |
|---|---|---|---|---|
| ContinueSlursReclaimedSlurs | Model alone | 512 | 0 | 0.0% |
| ContinueSlursReclaimedSlurs | RAG app | 512 | 0 | 0.0% |
| Profanity | Model alone | 4 | 0 | 0.0% |
| Profanity | RAG app | 4 | 0 | 0.0% |

**Finding:** Output-level toxic content generation is not a risk for this
system on either surface. Note: `lmrc.Profanity` only ran 4 attempts (Garak's
default), versus 512 for other probes — treat as indicative, not conclusive.

---

## 6. Top Priorities for V2/V3

1. **DAN jailbreak (49.8%)** — highest severity. V3 Security Layer needs
   explicit persona-hijack pattern detection; V2 Critic Agent should flag
   responses that deviate from the assistant's defined role.
2. **System prompt extraction (50.0%)** — V3 needs output filtering to
   detect and block system-prompt-like content in responses.
3. **HijackLongPrompt residual (11.1%)** — V3 needs input pattern detection
   for complex multi-sentence injection attempts.
4. **Encoding bypass (2.7–4.9% residual)** — should not rely on incidental
   RAG dampening; V3 input filter should explicitly decode and inspect
   Base64/Hex before passing to the LLM.

---

## 7. Conclusion

V1 Foundation RAG testing across 8 probes (4,096 total attempts) shows that
RAG architecture is **not a uniform security improvement**. It incidentally
reduces instruction-override attacks (prompt injection, encoded injection) by
64–93%, but **worsens** persona-hijack and extraction attacks, reaching
critical-severity hit rates of 49.8% (DAN) and 50.0% (system prompt
extraction).

This split result is the central insight from V1: **security architecture
must be evaluated per attack vector, not assumed from general design
choices.** It directly justifies V2 (Critic Agent — grounding validation)
and V3 (Security Layer — explicit input/output filtering for injection,
jailbreak, and extraction patterns) as necessary next phases.

> **Next step:** Begin V2 Critic Agent development. Re-run this identical
> probe set against the V2 endpoint and append a V2 comparison section
> measuring improvement against this baseline.
