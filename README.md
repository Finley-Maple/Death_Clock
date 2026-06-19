# Death Clock — Temporal Encoding for Mortality Risk Stratification

Code for the paper:

> **Temporal Structure for Mortality Risk Stratification: Evaluating Encoding Strategies Across LLM Paradigms on UK Biobank**
> [[manuscript repo]](https://github.com/Finley-Maple/Death_Clock_Draft)

We study how *temporal encoding* of electronic health records (ICD-10 trajectories, biomarkers, demographics) affects all-cause mortality prediction in a UK Biobank respiratory disease subgroup (n=124,158, age ≥60, event rate 26.2%). Five input-representation settings are evaluated under a uniform CoxPH survival head, with encoder architecture and fusion method as secondary ablations (12 cells total).

---

## Five-Setting Experiment Matrix

| Setting | Input representation | Encoder | Fusion |
|---------|----------------------|---------|--------|
| **S1** — Cross-sectional CoxPH | 39-dim numeric vector (biomarkers, demographics, comorbidities) | — | — |
| **S2** — Delphi zero-shot | Ordinal ICD token stream `44yr:J45 → 52yr:F32 → …` | Delphi (generative transformer) | — |
| **S3** — Isolated temporal prose | Natural-language sentences per event | Qwen3-8B / Bio_ClinicalBERT | concat / sum |
| **S4** — Decoupled time + event | 128-dim sinusoidal age encoding ∥ ICD token embedding, mean-pooled | Qwen3-8B / Bio_ClinicalBERT | concat / sum |
| **S5** — Unified summary prose | Single paragraph: demographics + lab values + disease history | Qwen3-8B / Bio_ClinicalBERT | none (embedding only) |

**Reference configuration** for S3/S4: Qwen3-8B + concatenation fusion.  
**Fusion (S3–S4):** concatenation = PCA(embedding → 64-dim) ∥ 39-dim raw → 103-dim CoxPH input; summation = PCA(embedding → 60-dim) + 60-dim standardised raw → 60-dim CoxPH input.

### Key results (test set)

| Setting | C-index | IBS | Mean TD-AUC |
|---------|---------|-----|-------------|
| S1 — Baseline CoxPH | 0.6191 | 0.1410 | 0.5999 |
| S2 — Delphi | 0.6161 | 0.1846 | 0.5984 |
| S3 — Prose (Qwen·concat) | 0.6156 | 0.1409 | 0.6074 |
| **S4 — Decoupled (Qwen·concat)** | **0.6233** | 0.1410 | 0.5991 |
| S5 — Unified prose (Qwen) | 0.6195 | **0.1408** | **0.6128** |

S4 vs S3 bootstrap ΔC = +0.008, 95% CI (+0.005, +0.010), p < 0.001.

---

## Directory Structure

```
├── embedding/                      # Embedding extraction (S3–S5)
│   ├── qwen_embedding.py           # Qwen3-Embedding extractor (S3, S5 texts; S4 tokens)
│   ├── clinical_bert_embedding.py  # Bio_ClinicalBERT extractor
│   └── trajectory_embedding.py     # Decoupled age+token pipeline (S4)
├── preprocessing/                  # Input serialisation
│   ├── natural_text_conversion.py  # EHR → prose sentences (S3/S5)
│   └── generate_trajectory_text.py # EHR → Delphi-style token stream (S2/S4)
├── evaluation/                     # Evaluation & metrics
│   ├── evaluate_benchmarking.py    # S1: CoxPH on raw features
│   ├── evaluate_delphi.py          # S2: Delphi zero-shot
│   ├── evaluate_embedding_survival.py  # S3–S5: embedding + CoxPH
│   ├── bootstrap_compare.py        # Paired bootstrap CIs and p-values
│   ├── unified_evaluation.py       # Aggregate all results
│   ├── plot_comparison.py          # Figure 2a: reference-config bar chart
│   ├── plot_task_illustration.py   # Figure 0: prediction task schematic
│   └── plot_age_encoding.py        # Sinusoidal age encoding visualisation
├── benchmarking/                   # Survival dataset preprocessing
├── Delphi/                         # Delphi model code (Shmatko et al. 2024)
├── scripts/                        # Batch embedding run scripts
├── data/                           # Raw & processed data (gitignored — UKB data)
└── manuscript/                     # LaTeX source (see Death_Clock_Draft repo)
```

> **Data privacy:** All UK Biobank participant data lives in `data/` which is gitignored. No eids or individual-level records are committed to this repository.

---

## Pipeline

### 0. UKB data extraction

Extract raw UK Biobank fields into `data/`. See `UKB_extraction/` for tooling.  
Required fields: `p41202`, `p41204`, `p41270` (hospital episode statistics); demographic and biomarker fields.

### 1. Build survival dataset

```bash
python benchmarking/preprocess_survival.py     # → data/preprocessed/autoprognosis_survival_dataset.csv
python benchmarking/preprocess_diagnosis.py    # → data/preprocessed/disease_before60_features.csv
```

### 2. Define shared cohort split (70 / 15 / 15, stratified)

```bash
python evaluation/cohort_split.py              # → evaluation/cohort_split.json
```

### 3. Serialise inputs

```bash
# S3 / S5 — prose sentences
python preprocessing/natural_text_conversion.py \
    --trajectory-csv data/preprocessed/disease_trajectory.csv \
    --output-csv     data/preprocessed/text_before60.csv

# S2 / S4 — age-tagged token stream
python preprocessing/generate_trajectory_text.py \
    --output-csv  data/preprocessed/trajectory_before60.csv
```

### 4. Compute embeddings (S3–S5)

```bash
# Qwen3-8B — prose texts (S3 / S5), GPU server
python embedding/qwen_embedding.py \
    --input-csv  data/preprocessed/text_before60.csv \
    --output-dir data/preprocessed/embeddings_text \
    --model-name Qwen/Qwen3-Embedding-8B

# Qwen3-8B — decoupled trajectory (S4)
python embedding/trajectory_embedding.py \
    --input-csv  data/preprocessed/trajectory_before60.csv \
    --output-dir data/preprocessed/embeddings_traj \
    --token-mode qwen \
    --model-name Qwen/Qwen3-Embedding-8B

# Bio_ClinicalBERT — prose texts (S3 / S5)
python embedding/qwen_embedding.py \      # calls clinical_bert_embedding internally via --encoder bcb
    ...                                   # or run clinical_bert_embedding.py directly
```

Batch scripts for all 12 cells are in `scripts/`.

### 5. Evaluate

```bash
# S1
python evaluation/evaluate_benchmarking.py

# S2
python evaluation/evaluate_delphi.py --split test --save-preds

# S3 / S4 (concat)
python evaluation/evaluate_embedding_survival.py \
    --embedding-dir data/preprocessed/embeddings_text \
    --method-name s3_qwen_concat \
    --baseline-mode all --fusion concat

# S5
python evaluation/evaluate_embedding_survival.py \
    --embedding-dir data/preprocessed/embeddings_text \
    --method-name s5_qwen \
    --baseline-mode none
```

### 6. Bootstrap confidence intervals

```bash
python evaluation/bootstrap_compare.py \
    --n-resamples 1000 --seed 42    # → evaluation/bootstrap_results.json
```

### 7. Unified comparison table

```bash
python evaluation/unified_evaluation.py    # → evaluation/unified_comparison.csv / .json
```

---

## Requirements

Python 3.9+, PyTorch ≥ 2.4.0.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Key packages:

| Package | Version | Used by |
|---------|---------|---------|
| `torch` | ≥2.4.0 | Delphi, Qwen3-Embedding |
| `transformers` | ≥4.53.0 | Qwen3-8B, Bio_ClinicalBERT |
| `lifelines` | ≥0.27.0 | CoxPH (S1, S3–S5) |
| `scikit-survival` | ≥0.22.0 | TD-AUC, IBS |
| `numpy` | <2.0 | All components |

> **Server note:** `torchvision` must be uninstalled on shared GPU servers to avoid import conflicts with `transformers`. The embedding scripts handle this automatically.

---

## Citation

If you use this code, please cite:

```
[BibTeX to be added upon publication]
```
