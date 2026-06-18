# Results Presentation Redesign

**Date:** 2026-06-18
**Scope:** Replace the heavy 12-run barplot (Figure 2) and two reference-config tables (Table 2, Table 3) with a full 12-cell results table and two focused barplots.

---

## Motivation

- Current Figure 2 (`figure_main`) plots all 12 experimental cells in a single barplot, making individual settings hard to distinguish.
- Table 2 (reference config results) and Table 3 (TD-AUC by horizon) present the same reference-configuration data in tabular form, redundantly alongside the figure.
- Mean TD-AUC and horizon-specific TD-AUC are separated across tables, obscuring their relationship.

---

## What Changes

| Old artefact | Action |
|---|---|
| Figure 2 (`figure_main`) | Retired entirely |
| Table 2 `tab:main` (reference config results) | Converted to Figure 2a (barplot) |
| Table 3 `tab:horizon` (TD-AUC by horizon) | Converted to Figure 2b (barplot) |
| — | New Full Table `tab:full` added (all 12 cells) |
| Tables 4–7 (bootstrap CIs, contrasts, encoder, fusion) | Unchanged |

---

## New Full Table (`tab:full`)

### Structure

- **12 rows** — one per experimental cell, grouped by setting with `\midrule` separators:
  - S1: Baseline CoxPH
  - S2: Delphi zero-shot
  - S3 × Qwen × concat / sum
  - S3 × BCB × concat / sum
  - S4 × Qwen × concat / sum
  - S4 × BCB × concat / sum
  - S5 × Qwen
  - S5 × BCB

- **9 columns** (in this order):
  `Setting | Encoder | Fusion | C-index | IBS | Mean TD-AUC | 9.5yr AUC | 15.4yr AUC | 19.7yr AUC`
  — C-index and IBS paired as discrimination/calibration; Mean TD-AUC and the three horizon columns contiguous so the temporal trend reads left-to-right.

- **Point estimates only** — bootstrap CIs remain in their own table (Table 4).

- **Formatting:**
  - `\resizebox{\textwidth}{!}{...}` wrapping a `\footnotesize` tabular to fit LNCS column width
  - Best value per column in **bold**
  - The 5 reference-configuration rows (S1, S2, S3·Qwen·concat, S4·Qwen·concat, S5·Qwen) lightly shaded (`\rowcolor{gray!15}`) to preserve the main-results reading path

### Data source

All values already present across `tab:main`, `tab:horizon`, `tab:encoder`, and `tab:fusion` in `experiments.tex` — no new computation required.

---

## Figure 2a — 3-Panel Barplot (replaces Table 2)

**File:** `evaluation/plot_comparison.py` (new function or new script `plot_results_bars.py`)
**Output:** `manuscript/figures/figure_2a.pdf` + `.png`

### Layout

- 3 panels side by side (1 row × 3 columns), shared y-tick style
- **Panel A:** C-index — 5 horizontal bars (reference configs S1–S5); dotted vertical reference line at S1 baseline value
- **Panel B:** Mean TD-AUC — same layout
- **Panel C:** IBS — same layout; label or downward arrow indicating "lower is better"

### Settings / bars (x-axis labels)

S1 Baseline | S2 Delphi | S3 Prose+Qwen | S4 Decoupled+Qwen | S5 Unified Qwen

### Style

- Consistent colour palette across Figure 2a and Figure 2b (one colour per setting)
- No error bars in Figure 2a (CIs are in Table 4)
- Figure width: `\textwidth`; height: ~3.5 inches

---

## Figure 2b — Grouped Horizon Barplot (replaces Table 3)

**File:** `evaluation/plot_comparison.py` (new function or new script `plot_horizon_bars.py`)
**Output:** `manuscript/figures/figure_2b.pdf` + `.png`

### Layout

- Single panel, grouped bar chart
- **X-axis:** 3 horizon groups — 9.5yr | 15.4yr | 19.7yr
- **Within each group:** 5 bars, one per reference-configuration setting, colour-coded as in Figure 2a
- **Y-axis:** TD-AUC (range approximately 0.50–0.66)
- Legend identifies the 5 settings

### Key visual story

Delphi (S2) rises from near-chance (~0.52) at 9.5yr to peak (~0.65) at 19.7yr — this inversion becomes immediately visible as its bar diverges from the cluster of CoxPH-based methods, which are comparatively flat across horizons.

---

## Manuscript Text Updates

- **experiments.tex:** Remove Table 2 (`tab:main`) and Table 3 (`tab:horizon`) environments; add Full Table (`tab:full`); add Figure 2a and Figure 2b environments with updated captions; update all `\ref{}` calls accordingly.
- **Narrative paragraphs:** Any sentence citing `Table~\ref{tab:main}` or `Table~\ref{tab:horizon}` redirects to the appropriate figure or the full table.
- The "Note on effect magnitudes" paragraph and Finding 3 paragraph currently reference both tables — update references after the new figures are in place.

---

## Files to Create / Modify

| File | Action |
|---|---|
| `evaluation/plot_results_bars.py` | New script producing Figure 2a |
| `evaluation/plot_horizon_bars.py` | New script producing Figure 2b |
| `manuscript/figures/figure_2a.pdf/.png` | New outputs |
| `manuscript/figures/figure_2b.pdf/.png` | New outputs |
| `manuscript/sections/experiments.tex` | Replace Table 2 + Table 3 with Full Table + Fig 2a + Fig 2b |

---

## Out of Scope

- No changes to Tables 4–7 (bootstrap CIs, contrasts, encoder ablation, fusion ablation)
- No changes to Figure 1 (pipeline overview)
- No new experiments or metric computation
