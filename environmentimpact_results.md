# Environmental Impact Results

Answers to the five environmental assessment questions for Task 2,
filled in with actual measured data where available.

---

## 1. Data required to estimate the carbon footprint

Four inputs are needed. All four were obtained automatically via CodeCarbon v3.2.6,
which reads hardware identifiers, samples power every 5 seconds (training) or 1 second
(inference), and applies the country grid intensity from its internal dataset.

### Hardware type and power envelope

| Scenario | GPU | CPU | RAM |
|----------|-----|-----|-----|
| Baseline (GPU) | NVIDIA GeForce RTX 3090 | Intel Core i7-10750H | 15.75 GB |
| Alt A (GPU) | NVIDIA GeForce RTX 3090 | Intel Core i7-10750H | 15.75 GB |
| Alt B (CPU) | not used (GTX 1650 Ti idle) | Intel Core i7-10750H | 15.75 GB |

### Wall-clock duration

| Scenario | Duration (s) |
|----------|-------------|
| Baseline (4/4/128, GPU) | 43.1 |
| Alt A (8/8/256, GPU) | 133.3 |
| Alt B (4/4/128, CPU) | 1324.9 |

### Actual power under load (from CodeCarbon)

| Scenario | GPU power (W) | CPU power (W) | RAM power (W) |
|----------|--------------|--------------|--------------|
| Baseline | 296.7 | 5.0 | 10.0 |
| Alt A | 353.9 | 4.7 | 10.0 |
| Alt B | 0.25 (idle) | 12.3 | 10.0 |

### Carbon intensity of the Danish electricity mix

CodeCarbon auto-detected Denmark (DNK) via IP geolocation and applied its internal
grid intensity. Back-calculated from results: 0.000568 kgCO2e / 0.003746 kWh =
**~152 gCO2e/kWh**, consistent with the Energinet 2024 annual average for Denmark.

---

## 2. CodeCarbon measurements

All values from `out/results_summary.csv`. Training figures are per full training run
(2000 iterations). Inference figures are per single response (averaged over 30 runs for
baseline/altA, 10 runs for altB).

### Training

| Scenario | Parameters | Energy (kWh) | CO2e (g) |
|----------|-----------|-------------|---------|
| Baseline (4/4/128, GPU) | 834,432 | 0.003746 | 0.568 |
| Alt A (8/8/256, GPU) | 6,400,768 | 0.013634 | 2.068 |
| Alt B (4/4/128, CPU) | 834,432 | 0.006655 | 1.009 |

### Inference (per response)

| Scenario | Tokens | Energy (Wh) | CO2e (mg) |
|----------|--------|------------|---------|
| Baseline (4/4/128) | 100 | 0.00754 | 1.14 |
| Alt A (8/8/256) | 100 | 0.01357 | 2.06 |
| Alt B (4/4/128) | 1000 | 0.06528 | 9.90 |

---

## 3. Comparative analysis

### Training

Alt A uses the most energy per run (0.014 kWh, 2.07 gCO2e), **3.6x the baseline**
(0.004 kWh, 0.57 gCO2e). This is expected: Alt A has 8x more parameters than baseline
(6.4M vs 0.83M), so each forward and backward pass requires more FLOPs. The GPU handles
it in 133 seconds vs 43 seconds for the baseline, but the higher GPU power draw (354W vs
297W) compounds the effect.

Alt B (same model as baseline but on CPU) uses 0.007 kWh, **1.8x the baseline**.
Despite the CPU drawing far less power than the GPU (12W vs 297W), the 30x longer
runtime (1325s vs 43s) means total energy is higher. This shows that for this model size,
GPU is more energy-efficient per training run even though it draws more instantaneous power.

Energy ranking for training: **Alt A > Alt B > Baseline**

### Inference

Alt B (1000 tokens) uses 0.065 Wh per response, **8.7x the baseline** (0.0075 Wh).
Token count is the dominant driver: 10x more tokens produces ~8.7x more energy (slightly
sublinear due to fixed overheads at the start of each call).

Alt A (larger model, same 100 tokens as baseline) uses 0.014 Wh, **1.8x the baseline**.
A larger model means more computation per token even at the same output length.

Energy ranking for inference: **Alt B > Alt A > Baseline**

---

## 4. Safe operating space (grandfathering)

### Fixed values

| Value | Source | Figure |
|-------|--------|--------|
| Global planetary boundary | Bjorn et al. 2015, tool cell C6 | 6.81e12 kgCO2e/yr |
| World population | Tool cell C7 | 8.00e9 |
| PB per capita | C6/C7 | **851.5 kgCO2e/cap/yr** |
| Denmark per-capita emissions (F14) | "world carbon footprint data" sheet in Mobility Excel | **[PLACEHOLDER - look up DK row, ~10,000-11,000 kgCO2e/cap/yr]** |

### Formula (applied to each scenario)

```
F16 = F15 / F14          share of per-capita budget used by this FU
F17 = F16 * 851.5        sustainable boundary for this FU (kgCO2e)
F18 = F15 / F17          exceedance ratio
```

Note: under grandfathering, F18 simplifies to F14 / 851.5 for every scenario.
This means all scenarios have the same exceedance ratio, equal to the factor by which
Denmark's current per-capita emissions exceed the planetary boundary. If F14 = 10,500,
then F18 = 10,500 / 851.5 = **~12.3 for all scenarios**. The scenarios differ in their
absolute emissions and absolute sustainable boundaries, not in their exceedance ratios.

### Results table (fill in F14 from the Excel tool)

| Scenario | Emissions F15 (kg) | F14 (kg/cap/yr) | Boundary F17 (kg) | Exceedance F18 |
|----------|--------------------|-----------------|-------------------|----------------|
| Training Baseline | 0.000568 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Training Alt A | 0.002068 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Training Alt B | 0.001009 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Baseline | 0.00000114 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Alt A | 0.00000206 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Alt B | 0.00000990 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

Once you have F14, plug it in everywhere. F17 = (F15/F14) * 851.5. F18 = F14/851.5.

---

## 5. Economic allocation (optional)

Economic allocation assigns a sustainable boundary based on the economic value of the
activity relative to a person's total annual spending (GDP PPP per capita).

### Formula

```
I14 = Danish GDP per capita PPP in DKK (~432,647 DKK/cap/yr, World Bank)
I15 = cloud cost of one FU in DKK
I16 = I15 / I14
I17 = I16 * 851.5        sustainable boundary under EA (kgCO2e)
I18 = F15 / I17          exceedance under EA
```

### Cloud cost estimates (AWS on-demand, retrieved [PLACEHOLDER date])

GPU scenarios mapped to `g4dn.xlarge` (NVIDIA T4, $0.526/hr).
CPU scenario mapped to `c6i.large` ($0.085/hr).
USD to DKK conversion rate: [PLACEHOLDER - check on day of submission, ~6.9 DKK/USD].

| Scenario | Duration (s) | Rate (USD/hr) | Cost (USD) | Cost (DKK) |
|----------|-------------|--------------|-----------|-----------|
| Training Baseline | 43.1 | 0.526 | 0.00630 | [PLACEHOLDER] |
| Training Alt A | 133.3 | 0.526 | 0.01948 | [PLACEHOLDER] |
| Training Alt B | 1324.9 | 0.085 | 0.03128 | [PLACEHOLDER] |
| Inference Baseline | [PLACEHOLDER] | 0.526 | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Alt A | [PLACEHOLDER] | 0.526 | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Alt B | [PLACEHOLDER] | 0.526 | [PLACEHOLDER] | [PLACEHOLDER] |

### EA results table

| Scenario | Emissions F15 (kg) | Cost I15 (DKK) | I16 | Boundary I17 (kg) | Exceedance I18 |
|----------|--------------------|---------------|-----|-------------------|----------------|
| Training Baseline | 0.000568 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Training Alt A | 0.002068 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Training Alt B | 0.001009 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Baseline | 0.00000114 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Alt A | 0.00000206 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Inference Alt B | 0.00000990 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

### GF vs EA comparison

Under grandfathering the budget scales with each scenario's own emissions, giving the same
exceedance ratio for all scenarios (~12-13 for Denmark). Under economic allocation the
budget is proportional to cost: since cloud compute is extremely cheap relative to a
person's annual income (I16 is tiny), I17 is far smaller than the GF boundary, and
exceedance ratios under EA will be much higher than under GF. This shows that when
judged by economic weight, AI compute is very carbon-intensive per unit of economic value.
