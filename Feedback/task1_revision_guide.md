# Task 1 Revision Guide

Sources: peer feedback (8 reviewers) + cross-reference with what was actually done in Task 2.

Edits are grouped by section. Each entry states the **location**, the **problem**, and the **exact fix**.

---

## Section 1.1 — Motivation and Context

### 1. No citations for factual claims (raised by Magnus Evensen, Amal Almis)

The introduction makes several specific claims without any source:
- "ChatGPT3 in November 2022"
- "ChatGPT 5 (the latest model) receives as much as 2.5 billion prompts a day"
- "the average energy-consumption of a prompt is 18 watt-hours"
- "typical modern nuclear power plants produce between 1 and 1.6 GW"

**Fix:** Add inline citations for each claim and add a References section at the end of the document. The Task 2 submission already includes a reference for the carbon footprint methodology paper (Jeanquartier et al., 2022) — this can serve as a template for the citation format. Each factual claim in the introduction needs a numbered inline citation.

---

## Section 1.2 — Object of Assessment

### 2. FU 1.2.1 (Training): Reference says "the table above" but the table is below

The training FU text reads:
> "...according to the fixed training procedure summarized in the table above..."

Table 1 (Hyperparameter Configuration) appears **after** this sentence on the next page, not above it.

**Fix:** Replace "the table above" with "Table 1".

---

### 3. FU 1.2.1 (Training): Missing measurable performance condition (raised by Magnus Evensen)

Current text:
> "The functional unit for the training phase is defined as training one instance of the Small Language Model according to the fixed training procedure summarized in the table above, resulting in a trained checkpoint ready for prompting, in Denmark."

There is no measurable output condition. Without one, it is ambiguous what "ready for prompting" means.

**Fix:** Replace the above sentence with:
> "The functional unit for the training phase is defined as training one instance of the nanoGPT-based Small Language Model (834,432 parameters in the baseline configuration) for exactly 2,000 iterations according to the fixed procedure in Table 1, producing a saved model checkpoint on consumer hardware operating in Denmark."

This adds: model name, parameter count, iteration count as the fixed stopping criterion, and removes the vague "ready for prompting."

---

### 4. FU 1.2.1 (Training): Table 1 DEVICE row is misleading

Table 1 lists `DEVICE = cpu` with description "Primary compute device (CUDA compatible)."

This is inaccurate: the training baseline and Alt A were run on GPU (NVIDIA GeForce RTX 3090). Only Alt B used CPU. The table lists the code default, not what was actually used.

**Fix:** Change the DEVICE row in Table 1 to:

| Parameter | Value | Description |
|-----------|-------|-------------|
| DEVICE | GPU / CPU | Set per scenario (see Table 4); GPU = NVIDIA GeForce RTX 3090, CPU = Intel Core i7-10750H |

Or alternatively, remove DEVICE from Table 1 entirely and note it is scenario-specific, since Table 4 already handles this.

---

### 5. FU 1.2.2 (Inference): Token count is wrong

Current text:
> "The functional unit for the inference phase is defined as generating **200 new output token** in response to one user prompt..."

The actual runs used **100 tokens** for baseline and Alt A, and **1000 tokens** for Alt B. 200 does not match any scenario.

**Fix:** Replace "200 new output token" with "a fixed number of new output tokens (100 for Baseline and Alternative A; 1000 for Alternative B)". Alternatively, define the baseline FU as 100 tokens and note that Alt B varies the token count as its scenario parameter.

---

### 6. FU 1.2.2 (Inference): Broken table reference

Current text ends with:
> "...summarized in Table ??, in Denmark."

**Fix:** Replace "Table ??" with "Table 2".

---

### 7. FU 1.2.2 (Inference): Does not mention checkpoint or hardware (raised by Marcus Christoffersen, Magnus Evensen, Christian Byskov)

The inference FU does not say that inference requires a trained model checkpoint as input, and does not specify hardware.

**Fix:** Replace the current inference FU sentence with:
> "The functional unit for the inference phase is defined as generating a fixed number of new output tokens in response to the prompt 'To be, or not to be', using a trained model checkpoint loaded onto a consumer GPU (NVIDIA GeForce RTX 3090) in Denmark, following the fixed inference procedure in Table 2."

---

### 8. FU 1.2.2 (Inference): Table 2 DEVICE row is wrong

Table 2 lists `DEVICE = cpu` with description "Automatic hardware acceleration detection."

The actual inference for all three scenarios ran on **GPU (NVIDIA GeForce RTX 3090)**, confirmed by CodeCarbon output (gpu_utilization: 32–36%, gpu_power: 37–44W). The CPU was tracked but not used for computation.

**Fix:** Change the DEVICE row in Table 2 to:

| Parameter | Value | Description |
|-----------|-------|-------------|
| DEVICE | cuda | NVIDIA GeForce RTX 3090; auto-detected via `torch.cuda.is_available()` |

---

### 9. Object of Assessment: Specify the model (raised by Kaja Hovinbøle, Marcus Christoffersen, Christian Byskov)

The section introduces "a Small Language Model" without naming it.

**Fix:** In the opening paragraph of 1.2, after "A Small Language Model (SLM) is the little brother of the LLMs...", add a sentence such as:
> "Specifically, we use a character-level language model based on the nanoGPT architecture (Karpathy, 2023), trained on the Tiny Shakespeare dataset (~1.1M characters)."

This also addresses Anna Christiansen's suggestion to mention dataset size.

---

## Section 1.3 — System Boundaries

### 10. Exclusions not categorised as "limited relevance" vs "simplification" (raised by Marcus Christoffersen, Christian Byskov, Carlos Lucero)

Current text excludes hardware construction, raw materials, etc. with one generic justification: "these processes would significantly expand the scope and complexity beyond the defined functional units."

The case study description asks you to distinguish between exclusions because they are **truly irrelevant** and exclusions made purely for **practical simplification**.

**Fix:** Replace the exclusion paragraph with two explicit categories:

> **Excluded due to limited relevance to the functional unit:** The use stage beyond the defined inference prompts (e.g., user interface, networking) and intermediate data storage are excluded because their energy contribution relative to the compute-heavy training and inference operations is negligible at the scale of this study.
>
> **Excluded for simplification:** The manufacturing and disposal of computing hardware (GPU, CPU, RAM) are excluded for practical reasons. While embodied carbon in hardware is a real impact, quantifying it requires hardware-specific lifecycle inventory data that is outside the scope of this study. This is a recognised limitation.

---

### 11. No system boundary diagram (raised by Christian Byskov)

**Fix:** Add a simple box-and-arrow diagram showing the lifecycle stages (Technical Design → Data Preparation → Training → Inference/Testing) with a clear boundary line indicating what is inside (electricity consumption during training and inference) and outside (hardware manufacturing, disposal, data centre infrastructure).

---

### 12. Danish energy mix not elaborated (raised by Christian Byskov)

The text mentions "assuming operation in Denmark" but does not explain why geography matters.

**Fix:** After the sentence "Electricity consumption during both training and inference is included, assuming operation in Denmark," add:
> "Denmark's electricity grid is predominantly wind-powered, resulting in a carbon intensity of approximately 152 gCO2e/kWh (Energinet 2024 annual average, as applied by CodeCarbon). This is significantly lower than the global average (~494 gCO2e/kWh), meaning the same computation produces far less CO2e than in a coal-heavy grid. Comparing models across geographies would require grid-intensity normalisation, which is outside the scope of this study."

---

## Section 1.4 — Scenarios

### 13. Specific hardware models not stated anywhere in Task 1 (cross-reference with Task 2)

Task 2 Table 1 names the actual hardware used — NVIDIA GeForce RTX 3090 (GPU scenarios) and Intel Core i7-10750H (CPU training, Alt B) — but Task 1 never specifies these.

**No Hardware column is needed in Table 5 (inference scenarios).** Table 2 (Inference FU) already fixes the hardware for all inference scenarios once DEVICE is corrected to `cuda` — all three inference scenarios ran on the same GPU, so a column repeating "RTX 3090" three times would be redundant by definition.

The CPU hardware (Intel Core i7-10750H) belongs in the **training** context. Table 4 already has a Hardware column (GPU/GPU/CPU), so the specific model names should appear as a note under that table.

**Fix — add a note under Table 4:**
> *Hardware specifications: GPU = NVIDIA GeForce RTX 3090 (16 GB VRAM); CPU = Intel Core i7-10750H (12 threads, 2.6 GHz).*

**Fix — update Table 2 DEVICE description** (see item 8 above) to name the GPU explicitly, e.g. description: "NVIDIA GeForce RTX 3090; auto-detected via `torch.cuda.is_available()`". This is where the GPU model is formally stated for inference.

---

### 14. Scenarios lack expected sustainability impact reasoning (raised by Marcus Christoffersen, Nicolas Faynot, Kaja Hovinbøle)

The scenario descriptions explain what parameter changes but not what sustainability impact is expected and why.

**Fix — Training scenarios paragraph:** After "This isolates the effect of compute efficiency and runtime on total energy consumption," add:
> "We expect Alt B to use less instantaneous power than the GPU scenarios but to run for much longer, making the total energy outcome uncertain and interesting to measure."

**Fix — Inference scenarios paragraph:** After "This isolates the effect of response length on computation and energy use," add:
> "We expect Alt B to use significantly more energy per response due to the 10× token count, and Alt A to use more energy than the baseline due to the larger model, even at equal token count. These predictions can be directly tested against the CodeCarbon measurements in Task 2."

---

### 15. Scenario section intro: Add explicit link to FU (raised by Nicolas Faynot)

Current opening:
> "To evaluate how engineering and usage choices influence sustainability outcomes, we define one baseline scenario and two alternative scenarios for both the training and inference phases."

**Fix:** Add after this sentence:
> "Each scenario is evaluated against its respective functional unit, so that all differences in measured impact are caused solely by the varied parameter and not by differences in scope."

---

## Additional: Parameter count in Object of Assessment (raised by Marcus Christoffersen)

The FU section never states how many parameters the models have. Task 2 Table 2 lists these (834,432 for baseline, 6,400,768 for Alt A).

**Fix:** Add a sentence to section 1.2 or 1.4.1 such as:
> "The baseline model (4/4/128) has 834,432 parameters. Alternative A (8/8/256) has 6,400,768 parameters — approximately 7.7× more. Alternative B uses the same architecture as the baseline."

This should appear in the Object of Assessment or at the top of the Scenarios section so the scale of the models is clear before the scenarios are defined.

---

## Summary of Errors vs Stylistic Improvements

| # | Type | Section | Issue |
|---|------|---------|-------|
| 2 | Error | 1.2.1 FU | "table above" → Table 1 |
| 5 | Error | 1.2.2 FU | Token count says 200, should be 100 (baseline FU) |
| 6 | Error | 1.2.2 FU | "Table ??" broken reference → Table 2 |
| 4 | Error | Table 1 | DEVICE = cpu, but training baseline/altA used GPU |
| 8 | Error | Table 2 | DEVICE = cpu, but all inference ran on GPU |
| 1 | Improvement | 1.1 | Add citations |
| 3 | Improvement | 1.2.1 FU | Add parameter count and measurable stopping criterion |
| 7 | Improvement | 1.2.2 FU | Add checkpoint and hardware to FU definition |
| 9 | Improvement | 1.2 | Name the model (nanoGPT) and dataset size |
| 10 | Improvement | 1.3 | Distinguish relevance vs simplification exclusions |
| 11 | Improvement | 1.3 | Add system boundary diagram |
| 12 | Improvement | 1.3 | Elaborate on Danish energy mix |
| 13 | Improvement | Table 4 + Table 2 | Add specific hardware model names as note under Table 4; name GPU in Table 2 DEVICE description. No Hardware column needed in Table 5 (FU already fixes it). |
| 14 | Improvement | 1.4 | Connect scenario reasoning to expected sustainability impacts |
| 15 | Improvement | 1.4 | Add explicit link back to FU in scenario intro |
| — | Improvement | 1.2/1.4 | State parameter counts for all model configurations |
