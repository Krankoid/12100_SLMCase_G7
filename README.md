# Sustainability Assessment of a Small Language Model (SLM)

Group 7 repository for the case study **"Assessing the Sustainability of Language Models"**,
developed for the course **1210X Quantitative Methods to Assess Sustainability**, DTU, Spring 2026.

Based on the starter code provided by the course TAs
([CarlosFdHL/GWPinLanguageModels](https://github.com/CarlosFdHL/GWPinLanguageModels)).

## Repository structure

```text
.
├── data/
│   └── prepare.py                      # Dataset preparation (Tiny Shakespeare, char-level)
│
├── src/
│   ├── model.py                        # Model architecture
│   ├── train_og.py                     # Original training script
│   ├── prompt_og.py                    # Original inference script
│   │
│   ├── training_scenarios/             # CodeCarbon training scenarios
│   │   ├── scenario1_hardware_cpu.py
│   │   ├── scenario1_hardware_gpu.py
│   │   ├── scenario1_hardware.py
│   │   ├── scenario2_layers.py
│   │   └── scenario3_batchsize.py
│   │
│   └── inference_scenarios/            # CodeCarbon inference scenarios
│       ├── scenario4_temperature.py
│       └── scenario5_tokens.py
│
├── task3/
│   ├── ecologits_benchmark.py          # EcoLogits API-model benchmark
│   ├── ecologits_benchmark.csv         # EcoLogits results
│   └── figures/                        # Plotting scripts and output figures
│
├── out_lifecycle/                      # CodeCarbon outputs + checkpoints per scenario
│
├── env_requirements.yaml               # Conda environment
└── README.md
```

## Setup

```bash
conda env create -f env_requirements.yaml
conda activate slm-sustainability
```

## Running the code

### 1. Prepare the dataset

```bash
python data/prepare.py
```

### 2. Train / Inference (original scripts)

```bash
python src/train_og.py
python src/prompt_og.py
```

### 3. CodeCarbon scenarios

Each scenario script runs training or inference while logging energy and CO2e
emissions with CodeCarbon. Outputs are written to `out_lifecycle/<scenario>/`.

**Training scenarios:**

```bash
python src/training_scenarios/scenario1_hardware.py     # CPU vs GPU
python src/training_scenarios/scenario2_layers.py       # model size sweep
python src/training_scenarios/scenario3_batchsize.py    # batch size sweep
```

**Inference scenarios:**

```bash
python src/inference_scenarios/scenario4_temperature.py # sampling temperature
python src/inference_scenarios/scenario5_tokens.py      # generated token count
```

### 4. EcoLogits benchmark

Benchmarks impacts of large hosted API models (separate from the local SLM).

```bash
python task3/ecologits_benchmark.py
```

Results are written to `task3/ecologits_benchmark.csv`.
