# Design: Delphi Survival Curve Fix + Biomarker Text + Fusion

**Date:** 2026-06-07  
**Deadline:** ~2 weeks (manuscript submission)  
**Status:** Approved

---

## Context

Current results on full 124k cohort (18,623-row test set):

| Method | Test C-index | Test TD-AUC | IBS |
|--------|-------------|------------|-----|
| Baseline CoxPH (39 feat) | 0.6191 | 0.5999 | 0.1410 |
| Disease text + CoxPH | 0.6156 | 0.6074 | 0.1409 |
| Trajectory + CoxPH | 0.6238 | 0.6001 | 0.1410 |
| Delphi (zero-shot) | 0.5882 | 0.5318 | — |

Two problems: (1) Delphi TD-AUC at 9.5y is 0.463 (sub-chance) — broken aggregation. (2) Embedding methods plateau; biomarker signal lives only in the structured baseline, not in the text.

---

## Goals

1. Fix Delphi to produce a proper survival curve, enabling TD-AUC at all horizons and IBS.
2. Add a biomarker-augmented text embedding method (standalone, no structured baseline).
3. Add a late-fusion method combining disease text + trajectory embeddings with the raw structured baseline.

All survival heads remain CoxPH for cross-method comparability.

---

## Workstream 1 — Delphi Survival Curve

### Problem

`build_risk_scores` in `evaluation/evaluate_delphi.py:85-92` uses `logits.max(axis=(1,2))`, then was patched to `logsumexp(tail[-10:])`. Both produce a scalar with no monotone relationship to cumulative hazard at specific horizons. TD-AUC at 9.5y = 0.463 (worse than random).

### Design

Replace `build_risk_scores` with `build_survival_curve` that produces a proper `(n_patients, n_horizons)` survival probability matrix:

**Step 1 — Extract per-position hazard**
```
logits: (n_patients, seq_len, n_death_tokens)
death_logit[t] = mean(logits[:, t, :], axis=-1)   # (n_patients, seq_len)
hazard[t] = sigmoid(death_logit[t])               # per-step probability of death
```

**Step 2 — Accumulate survival function**
```
S[t] = ∏_{i=0}^{t} (1 - hazard[i])              # (n_patients, seq_len)
```

**Step 3 — Map sequence positions to calendar time**

Each position in the Delphi sequence has an associated age (encoded in the `.bin` file's age tensor `a`). Convert to days-since-60th-birthday:
```
time_days[t] = (age_at_position[t] - 60.0) * 365.25
```
Positions with `time_days < 0` (before age 60, i.e. history) are excluded from the survival accumulation; the survival curve begins at position where `time_days ≥ 0`.

**Step 4 — Interpolate at evaluation horizons**

Horizons are `[3471, 5620, 7213]` days (shared with CoxPH methods, from `metrics.derive_time_horizons`). For each patient, linearly interpolate `S(t)` at each horizon using the time grid from Step 3.

**Step 5 — Risk score**

For C-index: `risk_score = 1 - S(max_horizon)` — risk of dying before the longest horizon.  
For TD-AUC per horizon `τ`: `risk_at_τ = 1 - S(τ)`.

### Files modified

- `evaluation/evaluate_delphi.py`
  - Replace `build_risk_scores(logits)` → `build_survival_curve(logits, age_tensor, horizons)` returning `(survival_probs, risk_scores)`
  - `run_inference_on_bin` returns both logits and the age tensor `a` (already available in the batch)
  - Pass `survival_probs` to `metrics.compute_metrics` (same call signature as CoxPH methods)
  - Add `--splits` flag to loop over `{train, val, test}` so val metrics are populated in unified comparison

- `evaluation/unified_evaluation.py`
  - Parse `ibs` and `c_index_ipcw` from Delphi results (currently hard-coded as `null`)

### Edge cases

- Patients with no future positions (all history, no post-60 sequence): assign `S(t) = 1` for all horizons (censored immediately, excluded from TD-AUC pairs).
- Horizons beyond the last sequence position: extrapolate as `S(t) = S(last_position)` (flat survival after last observation).

---

## Workstream 2a — Biomarker-Augmented Text (Method 4)

### Design

**Survival head:** CoxPH on embedding only — **no structured baseline** (`--baseline-mode none`).  
Rationale: biomarker values appear in the text; including raw numerics alongside would double-count them and obscure whether the embedding representation carries the signal.

**Text format per patient:**
```
Demographics: 62-year-old male, BMI 28.4, current smoker, lives alone.
Lab results (age ~60): HbA1c 42 mmol/mol, total cholesterol 5.6 mmol/L,
  HDL 1.1 mmol/L, creatinine 95 μmol/L, CRP 3.2 mg/L, IGF-1 18.2 nmol/L,
  triglycerides 1.8 mmol/L, glucose 5.1 mmol/L, albumin 44 g/L, [...]
Disease history before age 60:
  At age 19.5, patient was diagnosed with J30 vasomotor and allergic rhinitis.
  At age 44.0, patient was diagnosed with N84 polyp of female genital tract.
  [No diseases diagnosed before age 60.] (if none)
```

**Implementation in `preprocessing/natural_text_conversion.py`:**

Add `--mode biomarker-before60` that calls:
1. `_create_demographics_text(patient_data)` — age, sex, BMI, smoking, alcohol, living alone
2. `_create_clinical_measurements_text(patient_data)` — all 27 lab biomarkers
3. `DiseaseBefore60TextConverter.convert_row(row)` — disease events before 60

Concatenate with newlines. Output: `data/preprocessed/biomarker_text_before60.csv` with columns `(eid, text)`. Filter to cohort eids from `cohort_split.json` (same as existing `--mode before60`).

**Embedding:** `qwen_embedding.py --input-csv biomarker_text_before60.csv --tag biomarker --model-name Qwen3-Embedding-8B` on featurize server. Output: `data/preprocessed/embeddings_biomarker/biomarker_embeddings.npz`.

**Evaluation:**
```bash
python3 evaluation/evaluate_embedding_survival.py \
    --embedding-dir data/preprocessed/embeddings_biomarker \
    --tag biomarker --method-name biomarker_text \
    --baseline-mode none \
    --pca-components 128
```
No changes to `evaluate_embedding_survival.py` needed.

---

## Workstream 2b — Late Fusion (Method 5)

### Design

Combine disease-history embeddings with raw structured features. Biomarker text embedding is **excluded** from the fusion (it would be redundant with the 39 structured baseline features and muddy the story).

**Input matrix:**
```
[disease_text_pca(128) | trajectory_pca(128) | baseline_scaled(39+indicators)]
total width: 256 + 39 + n_missing_indicators ≈ 300
```

PCA is applied **independently** per embedding block (fit on train, applied to val/test), producing unit-variance components. The structured baseline goes through the existing `_median_impute` + `standardize_features` path. The full combined matrix goes directly to CoxPH (no second PCA pass — `emb_block_width = 256` so the existing split-PCA logic in `run_cox_evaluation` applies PCA only to the embedding block, leaving structured features intact).

**New file: `evaluation/evaluate_fusion.py`**

```python
# Pseudocode
embeddings_text = load_embeddings(text_dir, "patient")      # {eid: 4096-d}
embeddings_traj = load_embeddings(traj_dir, "trajectory")   # {eid: 1152-d}
matrices = load_survival_matrices(baseline_cfg)             # 39+indicator features

# PCA per embedding, fit on train
text_pca  = fit_pca(train_embs_text,  n=128)
traj_pca  = fit_pca(train_embs_traj,  n=128)

# Merge: emb_block = [text_pca | traj_pca], then concat baseline
for split in [train, val, test]:
    emb = concat(text_pca.transform(split), traj_pca.transform(split))  # 256-d
    combined = merge_with_embeddings(baseline_matrix[split], {eid: emb})

cox_cfg = CoxConfig(pca_components=None, emb_block_width=256)
# emb_block_width=256 with pca_components=None → no further PCA, baseline kept raw
results = run_cox_evaluation(combined_matrices, cox_cfg, method_name="fusion")
```

Reuses: `load_survival_matrices`, `merge_with_embeddings`, `run_cox_evaluation`, `standardize_features` from `evaluation/survival_eval.py` — no changes to that file.

**CLI:**
```bash
python3 evaluation/evaluate_fusion.py \
    --text-embedding-dir data/preprocessed/embeddings_text \
    --traj-embedding-dir data/preprocessed/embeddings_traj \
    --text-pca-components 128 \
    --traj-pca-components 128 \
    --method-name fusion \
    --output-dir evaluation/fusion_results
```

---

## Final methods table

| # | Method | Baseline included | Expected C-index |
|---|--------|------------------|-----------------|
| 1 | Baseline CoxPH | Yes (39 feat) | 0.619 (fixed) |
| 2 | Disease text + CoxPH | Yes (39 feat) | 0.616 (fixed) |
| 3 | Trajectory + CoxPH | Yes (39 feat) | 0.624 (fixed) |
| 4 | **Biomarker text + CoxPH** | **No** | ~0.610–0.625 |
| 5 | **Fusion (text+traj emb + baseline)** | Yes (39 feat) | ~0.625–0.638 |
| 6 | Delphi (survival curve) | — (zero-shot) | C~0.59, TD-AUC TBD |

---

## Files to create / modify

| File | Action | Notes |
|------|--------|-------|
| `evaluation/evaluate_delphi.py` | Modify | Replace risk-score with survival curve; add multi-split loop |
| `evaluation/unified_evaluation.py` | Modify | Parse Delphi IBS + val metrics |
| `preprocessing/natural_text_conversion.py` | Modify | Add `--mode biomarker-before60` |
| `evaluation/evaluate_fusion.py` | **Create** | New script; reuses survival_eval internals |

**Not modified:** `evaluation/survival_eval.py`, `evaluation/metrics.py`, `embedding/qwen_embedding.py`, `evaluation/evaluate_embedding_survival.py` — all reused as-is.

---

## Testing

1. **Delphi smoke test (local):** Run `evaluate_delphi.py` with a small `.bin` slice; verify `survival_probs` shape is `(n_patients, n_horizons)`, all values in (0,1), S decreasing with time.
2. **Biomarker text spot-check:** Print 3 sample rows from `biomarker_text_before60.csv`; confirm demographics + labs + disease history all present; confirm no post-60 diseases appear.
3. **Fusion alignment:** Verify `combined.eids` for train/val/test match cohort split exactly (no silent drops from missing embeddings).
4. **Regression:** After all changes, `benchmarking_results.json` C-index should remain 0.6191 ± 0.0001 (the baseline pipeline is untouched).

---

## Implementation order

1. Delphi survival curve (`evaluate_delphi.py`) — isolated, testable locally with existing `.bin`
2. Biomarker text generation (`natural_text_conversion.py`) — CPU, runs locally
3. Server: embed biomarker text (Qwen3-8B, ~2h)
4. Server: evaluate Method 4 (`evaluate_embedding_survival.py --baseline-mode none`)
5. `evaluate_fusion.py` — new script, reuses existing building blocks
6. Server: evaluate Method 5 (fusion)
7. Re-run `unified_evaluation.py` → final six-method table
