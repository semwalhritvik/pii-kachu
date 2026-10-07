# PII-kachu: Local PII Detection, Verification, and Routing Pipeline
## 1. Project Objective
- Build a local data-sanitization pipeline that processes institutional documents through a lightweight NER system (e.g., GLiNER) to detect and redact sensitive Personally Identifiable Information (PII) at the edge.
Implement a local LLM-based verification step to evaluate the redacted text, serving as a quality-assurance cascade.
- Use the local LLM to create a feedback loop: if the text is over-redacted (e.g., masking famous historical figures alongside a student's name), the LLM highlights the errors and routes the text back to the NER system for correction.
- Ensure the local LLM acts as a strict privacy guard, verifying that no genuinely sensitive information bypasses the filters.
- Pseudonymize the successfully redacted data using simple ML models or rule-based replacement systems to preserve data utility for downstream processing.
- Execute a final privacy-routing decision: if the text can be safely masked and pseudonymized, it proceeds to a dynamic complexity router for cloud-based LLM processing; if the data cannot be safely masked, it hits a strict privacy boundary and is routed exclusively to an on-premise institutional LLM.

## 2. System Architecture
```
[Raw Institutional Documents]
               │
               ▼
 ┌────────────────────────────────────────────────┐
 │     Stage 1: Local Privacy Detection           │
 │     (Fast NER · Regex · GLiNER)                │
 └──────────────────────────┬─────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────┐
 │     Local LLM Verifier (Quality Check)         │
 │     · Over-redaction check  (e.g. History)     │
 │     · Under-redaction check (Leakage)          │
 └────┬───────────────────────────────────────────┘
      │ Needs Correction                │ Approved
      ▼                                 ▼
[Feedback Loop]              ┌──────────────────────┐
Back to NER Stage            │    Sensitive PII?    │
                             └──────┬────────────┬──┘
                                    │ Yes        │ No
                                    ▼            │
                       ┌────────────────────┐    │
                       │  Safely maskable & │    │
                       │  pseudonymizable?  │    │
                       └────┬──────────┬────┘    │
                            │ Yes      │ No      │
                            ▼          ▼         │
                 ┌──────────────┐ ┌──────────┐   │
                 │ Pseudonymize │ │  Strict  │   │
                 │ (name →      │ │ Boundary │   │
                 │   token)     │ │  (Zero   │   │
                 │              │ │  Egress) │   │
                 └──────┬───────┘ └────┬─────┘   │
                        └──────┬───────┘         │
                               └─────────────────┘
                                        │
                                        ▼
 ┌──────────────────────────────────────────────────────┐
 │          Stage 2: Context & Complexity Router        │
 │    (Privacy Boundary · Task Difficulty · Cost)       │
 └─────────────────────────────┬────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
        ┌─────────────────────┐  ┌─────────────────────┐
        │  Institutional LLM  │  │   Cloud Endpoint    │
        │  · Strict data      │  │   · Public/scrubbed │
        │  · Proprietary      │  │   · Low-risk,       │
        │    records          │  │     high-reasoning  │
        └─────────────────────┘  └─────────────────────┘
```

[Architecture based on the provided Stage 1 and Stage 2 multi-agent/routing framework logic]
## 3. Implementation Roadmap
### Phase 1 - Baseline Testing & Document Ingestion
Explore the proposed dataset and generate synthetic document examples containing varied entity types and contexts.
Run baseline local extraction (e.g., Microsoft Presidio) on fictional text to map out baseline strengths and weaknesses in standard entity detection.
Finalize the PII schema, ensuring it differentiates between strict personal identifiers and general contextual names (e.g., historical figures).
Deliverable: A working ingestion prototype and baseline NER redaction pipeline.
### Phase 2 - Implement GLiNER, LLM Verification & Pseudonymization
Implement a learned NER approach (GLiNER) for highly accurate, zero-shot entity detection.
Develop the local LLM verification cascade to audit the NER output, returning over-redacted or under-redacted text back to the NER layer for correction.
Build the pseudonymization engine using rule-based logic or lightweight ML to replace redacted spans with deterministic, reversible tokens.
Deliverable: A functioning multi-stage pipeline featuring NER detection, LLM-based self-verification, and pseudonymization.
### Phase 3 - The Routing Gate & Model Evaluation
Implement the "Yes/No" routing logic that evaluates whether the pseudonymized text can proceed to the cloud or must hit the strict local boundary.
Run an evaluation on a human-labeled, held-out test set, employing error analysis on both the NER detection step and the LLM verifier's accuracy.
Analyze and compare aggregate performance results, utilizing metrics that penalize PII leakage heavily (e.g., entity-level F2 score) and track excessive-redaction rates.
Deliverable: Final report and presentation detailing the pipeline's privacy-utility trade-offs and routing efficiency.
