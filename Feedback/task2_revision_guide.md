# Task 2 Revision Guide

Sources: peer feedback (8 reviewers) cross-referenced against the Task 2 submission.

Edits are grouped by section. Each entry states the **location**, the **problem**, and the **exact fix**.

---

## Section 1 — Environmental Impacts: Data and Measurement Setup

### 1. "Duration and hardware utilization also contribute" (Andreas Rattenborg)

The opening paragraph lists four inputs needed to estimate carbon footprint: hardware type, duration, power draw, and grid carbon intensity. Andreas flags that duration and hardware utilization should also be explicitly mentioned in the comparative analysis, not just the setup. This is already implicit in the text but not stated up front as a driver.

**Fix:** Minor — acceptable as-is, or add "hardware utilization rate" explicitly to the list of four inputs: *"the hardware type and utilization rate, how long it runs, how much power it draws under load, and the carbon intensity of the electricity grid."*

---

### 2. Alt B inference run count stated incorrectly (factual error, cross-reference with CodeCarbon data)

The setup paragraph states:
> "each inference scenario was repeated 30 times (10 for Alternative B **since it operates on CPU**)"

Alternative B did **not** operate on CPU during inference. All three inference scenarios ran on GPU (NVIDIA GeForce RTX 3090), confirmed by the CodeCarbon CSVs (gpu_utilization: 31.9%, gpu_power: 36.6W for Alt B inference). The 10-run count for Alt B was due to its 1000-token length making each run ~3× longer, not because of CPU usage.

**Fix:** Replace the parenthetical with:
> "(10 for Alternative B due to its 10× longer output length per run)"

---

## Section 1 — Environmental Impacts: Safe Operating Space

### 3. Method is named incorrectly — FCE is not grandfathering (raised by Levente Murgas, Christian Byskov)

The section is titled "Safe operating space" and opens with *"we applied the grandfathering approach."* But the method used allocates each scenario's environmental budget based on its **share of global data centre spending (cost)**. That is **Final Consumption Expenditure (FCE) allocation**, not grandfathering.

Grandfathering would assign each scenario a budget proportional to its current share of **actual emissions** within the sector — i.e., if your scenario produces 1% of all data centre emissions today, it gets 1% of the planetary budget. FCE does this based on spending instead.

These are two distinct allocation methods. The submission uses FCE throughout but calls it grandfathering in every instance.

**Fix:** Replace every instance of "grandfathering approach" / "grandfathering" in this section with "FCE allocation" or "Final Consumption Expenditure (FCE) allocation." The formula, table, and numerical results are correct — only the name is wrong.

---

### 4. Conclusion of "absolute sustainability" is overstated (raised by Andreas Rattenborg, Yannick Christensen)

The conclusion states:
> "All three training scenarios are absolutely sustainable"
> "This indicates that SLMs are in fact a sustainable alternative to LLMs"

This overstates what the results show. The SR values of 0.0138 are only valid under the FCE allocation method with the chosen assumptions (2020 baseline year, 2030 Paris target, $333B global data centre revenue). A different allocation method — or different baseline assumptions — would yield different ratios.

**Fix:** Qualify the conclusion explicitly. Replace the above with something like:
> "Under the FCE allocation method applied here, all three training scenarios fall well within their allocated environmental budget (SR ≈ 0.014). However, this result is allocation-method-dependent: FCE ties each scenario's budget to its cost, so more expensive scenarios automatically receive a larger allowance. This means the SR is not a measure of absolute sustainability, but rather of sustainability relative to a cost-proportional share of the sector budget."

Then soften the final SLM conclusion to:
> "Under these assumptions, SLMs appear to be a low-impact alternative to LLMs for both training and inference, though the conclusion depends on the allocation method chosen."

---

### 5. Circular logic in FCE allocation not sufficiently reflected upon (raised by Yannick Christensen, Christian Byskov)

The submission identifies the circularity itself in the final paragraph:
> "A challenging consequence of using FCE... is that scenarios like training Alternative A and inference Alternative B — which have significantly higher CO2 emissions — still maintain relatively low sustainability ratios. This occurs because these scenarios are also more expensive to execute."

This is identified but not reflected upon in terms of what it means for the conclusions. All three training scenarios have identical SR = 0.0138, which makes FCE useless for distinguishing between them environmentally.

**Fix:** Add a sentence making the implication explicit:
> "As a result, FCE allocation cannot differentiate between scenarios that have the same cost structure regardless of their emissions — the environmental ranking (Alt A > Alt B > Baseline for training) is not reflected in the sustainability ratios at all. For comparative purposes, the absolute emissions figures in Tables 2 and 3 are more informative than the SR values."

---

## Section 2 — Economic Impacts

### 6. Inconsistent cost basis: AWS for training, household electricity for inference (raised by Christian Byskov)

Training costs are calculated using AWS EC2 On-Demand pricing (cloud rates). Inference costs are calculated using Danish household electricity (2.5 DKK/kWh, a local consumer rate). These are from different economic contexts and are not directly comparable in a single LCC.

**Fix — Option A (cloud framing):** Price inference using the same AWS instance types as training. GPU inference on `g4dn.xlarge` at 3.63 DKK/hr, CPU inference on `c6i.large` at 0.59 DKK/hr. Multiply measured duration per inference run by the hourly rate.

**Fix — Option B (local framing):** Price training using the same measured energy × Danish electricity rate instead of AWS pricing.

Option A is more consistent with the report's cloud deployment framing. Whichever option is chosen, add a sentence explaining the rationale for the chosen basis.

---

### 7. LCC plot shows only two fixed N values — no crossover point visible (raised by Vlad Burlacu)

Figure 1 shows stacked bars for N = 10,000 and N = 100,000. This makes it impossible to see at what prompt volume Alt B's higher per-response cost makes it more expensive overall than the baseline or Alt A.

**Fix:** Replace the bar chart with a line plot of LCC(N) = C_train + N · C_inf as a continuous function of N (e.g., N from 0 to 10^6). Plot all three scenarios on the same axes. The crossover points where one scenario overtakes another become immediately visible. This is a stronger visual and directly answers the question of when the token-count trade-off matters economically.

---

## Section 3 — Social Impacts and Human Health

### 8. No stakeholder mapping — impacts described generically (raised by Christian Byskov, Vlad Burlacu)

The section lists four impact types (noise, air, water, employment) but refers only to "workers," "locals," and "people in offices" as undifferentiated groups. The task asks for impacts tied to specific stakeholder groups.

**Fix:** Restructure each subsection to name specific stakeholders and their relationship to the impact. For example:

- **Noise pollution:** Data centre technicians (prolonged occupational exposure, >85 dBA threshold); residents within 500m of facilities (sleep disruption, 24/7 exposure)
- **Air pollution:** Local residents (respiratory and cardiovascular risk from diesel exhaust); municipal health authorities (regulatory burden)
- **Water scarcity:** Local communities dependent on the same water supply (reduced availability); agricultural users in the same watershed
- **Employment:** Local construction and IT workers (positive — job creation); global knowledge workers in writing, coding, translation (negative — displacement risk)

---

### 9. Missing: privacy and data risk as a social impact (raised by Vlad Burlacu)

The section covers physical impacts (noise, air, water) and labour but does not address the social risks of the information processed by SLMs/LLMs.

**Fix:** Add a brief subsection after "The social impact" paragraph:

> **Data privacy and information risk**
> SLMs and LLMs process user-submitted text, which may include sensitive personal, professional, or proprietary information. Stakeholders affected include individual users (risk of data retention or exposure), organisations (confidentiality of internal documents submitted as prompts), and regulators (obligation to enforce GDPR-compliant data handling). Unlike the physical impacts above, these risks are not mitigated by operating in Denmark — they depend on where model weights and inference infrastructure are hosted and under which jurisdiction.

---

### 10. Social impacts section is qualitative only — no quantitative metrics (raised by Vlad Burlacu)

The submission describes impacts descriptively ("3–7 million gallons of water per day") but does not connect these figures to the specific scenarios assessed.

**Fix:** Add at least one scenario-specific quantitative metric. Water consumption is the most tractable: CodeCarbon reports a `water_consumed` field (all values were 0.0 in the output, indicating local hardware without a WUE model applied). Acknowledge this limitation and provide a reference figure, e.g.:
> "Cloud data centres report a Water Usage Effectiveness (WUE) of approximately 1.5–2.0 L/kWh. At 0.003746 kWh for baseline training, this implies approximately 5–7 mL of water consumed — negligible at this scale, but proportionally significant at the scale of a deployed LLM receiving billions of daily prompts."

---

### 11. UNEP S-LCA framework not referenced (raised by Yannick Christensen)

The social impacts section has no methodological grounding.

**Fix:** Add one sentence at the opening of Section 3:
> "The following social impacts are assessed qualitatively, drawing on the stakeholder categories defined in the UNEP Guidelines for Social Life Cycle Assessment of Products (2020): workers, local community, and society."

This frames the section methodologically without requiring a full S-LCA.

---

## Cross-cutting Issues

### 12. Scenarios vary multiple parameters at once — limits sensitivity analysis (raised by Levente Murgas, Olivia Droob, Viktor Guijarro)

Levente points out that a proper sensitivity analysis would vary **one parameter at a time across a range of values** from the baseline, e.g.:

- Scenario 1: layers = 8, 16, 32 (heads and embedding fixed)
- Scenario 2: hardware = CPU vs GPU (architecture fixed)

Instead, the current Alt A changes both architecture size and embedding simultaneously (4→8 layers, 4→8 heads, 128→256 embedding), so it is impossible to isolate which architectural change drives the energy difference.

**Important note:** Re-running the experiments is out of scope for the revision. The fix is to **acknowledge this as a limitation** rather than redesign the study.

**Fix:** Add a limitations paragraph at the end of Section 1 (Environmental Impacts):
> "A limitation of the scenario design is that Alternative A varies multiple architectural parameters simultaneously (layers, heads, and embedding size), making it impossible to isolate the effect of each individual change. A more rigorous sensitivity analysis would vary one parameter at a time across a range of values. The current design nonetheless illustrates the direction and magnitude of the combined effect."

---

### 13. No cross-dimensional summary table (raised by Christian Byskov)

There is no table or paragraph that brings together the environmental, economic, and social findings per scenario in one place.

**Fix:** Add a summary table at the end of the document:

| Scenario | CO2e training (g) | CO2e inference/response (mg) | Training cost (DKK) | LCC at N=100k (DKK) | Key social risk |
|----------|------------------|------------------------------|--------------------|--------------------|----------------|
| Baseline (4/4/128, GPU) | 0.568 | 1.14 | 0.043 | ~75 | Standard GPU data centre impacts |
| Alt A (8/8/256, GPU) | 2.068 | 2.06 | 0.134 | ~135 | Higher energy → proportionally greater impacts |
| Alt B (4/4/128, GPU) | 1.009 | 9.90 | 0.213 | ~653 | Highest inference cost at scale |

Add a one-paragraph narrative below the table summarising the overall recommendation across all three dimensions.

---

### 14. Insufficient visual representation (raised by Olivia Droob, Viktor Guijarro)

The submission has one figure (the LCC bar chart). The CodeCarbon data supports several additional visualisations that already exist in the `task3/figures/` folder of the repository.

**Fix:** Consider adding:
- A grouped bar chart of energy per scenario for training vs inference side by side
- The existing `training_co2.pdf` and `inference_co2.pdf` figures from `task3/figures/` if not already included

---

## Summary of Errors vs Improvements

| # | Type | Section | Issue |
|---|------|---------|-------|
| 2 | Error | Env. setup | Alt B inference stated as CPU — it ran on GPU |
| 3 | Error | Safe operating space | Method called "grandfathering" — it is FCE allocation |
| 4 | Improvement | Safe operating space | "Absolutely sustainable" conclusion is overstated — qualify it |
| 5 | Improvement | Safe operating space | Circular logic identified but implications not drawn out |
| 6 | Improvement | Economic | Inconsistent cost basis — training AWS, inference household electricity |
| 7 | Improvement | Economic | LCC plot: replace two-bar chart with continuous N sensitivity curve |
| 8 | Improvement | Social | Add specific stakeholder mapping to each impact type |
| 9 | Improvement | Social | Add data privacy/information risk subsection |
| 10 | Improvement | Social | Add at least one scenario-specific quantitative metric (e.g. WUE water) |
| 11 | Improvement | Social | Reference UNEP S-LCA framework as methodological anchor |
| 12 | Improvement | Env. (limitation) | Acknowledge that Alt A varies multiple parameters — limits sensitivity |
| 13 | Improvement | End of document | Add cross-dimensional summary table |
| 14 | Improvement | Throughout | Add visualisations (figures already exist in task3/figures/) |
| 1 | Minor | Env. setup | Add "hardware utilization rate" to the four-input list |
