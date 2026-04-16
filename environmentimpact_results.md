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

Use the **"absolute sustainability" sheet** in the Mobility Excel tool (the file
`Exercise solutions, Mobility - Absolute sustainability 1210X QMAS.xlsx` from DTU Learn).

Steps:
1. Open the "absolute sustainability" sheet.
2. Enter your emissions value (F15) for each scenario - the `emissions_kg_total` or
   `emissions_kg_per_run` value from `results_summary.csv`.
3. The sheet reads Denmark's per-capita emissions from the "world carbon footprint data"
   tab automatically (F14, around 10,000-11,000 kgCO2e/cap/yr for DK).
4. Read off F17 (sustainable boundary for this FU) and F18 (exceedance ratio).

The planetary boundary is 6.81e12 kgCO2e/yr divided across 8e9 people = 851.5 kgCO2e/cap/yr.
Exceedance ratios well above 1 are expected and are the intended finding.

### Results (fill in from the Excel tool)

| Scenario | Emissions F15 (kg) | GF boundary F17 (kg) | Exceedance F18 |
|----------|--------------------|----------------------|----------------|
| Training Baseline | 0.000568 | [from Excel] | [from Excel] |
| Training Alt A | 0.002068 | [from Excel] | [from Excel] |
| Training Alt B | 0.001009 | [from Excel] | [from Excel] |
| Inference Baseline | 0.00000114 | [from Excel] | [from Excel] |
| Inference Alt A | 0.00000206 | [from Excel] | [from Excel] |
| Inference Alt B | 0.00000990 | [from Excel] | [from Excel] |

---

## 5. Economic allocation (optional)

If required, apply economic allocation using the same "absolute sustainability" sheet:
enter the cloud cost of one FU (in DKK) as I15, and Danish GDP PPP per capita as I14
(~432,000 DKK/cap/yr, World Bank). The sheet calculates I17 and I18.

Cloud cost estimates (AWS on-demand):
- GPU scenarios: `g4dn.xlarge` at $0.526/hr
- CPU scenario: `c6i.large` at $0.085/hr

Multiply duration (seconds) by rate (USD/hr) / 3600, then convert USD to DKK (~6.9 DKK/USD).

| Scenario | Duration (s) | Cost (USD) |
|----------|-------------|-----------|
| Training Baseline | 43.1 | 0.00630 |
| Training Alt A | 133.3 | 0.01948 |
| Training Alt B | 1324.9 | 0.03128 |
