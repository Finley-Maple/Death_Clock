"""
Delphi evaluation adapter aligned with the shared survival metrics.

Outputs:
  - Patient-level risk scores for the Death token (per cohort split).
  - Survival metrics (C-index, TD-AUC, IBS placeholder) using evaluation/metrics.py.
  - Optional legacy DeLong summary via evaluate_auc_pipeline for comparison.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Delphi"))

from evaluation import data_access, metrics, survival_eval  # noqa: E402
from model import Delphi, DelphiConfig  # noqa: E402
from utils import get_batch, get_p2i  # noqa: E402
from evaluate_auc import evaluate_auc_pipeline  # noqa: E402

COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
DELPHI_LABELS = PROJECT_ROOT / "Delphi" / "delphi_labels_chapters_colours_icd.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "delphi_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_death_tokens(labels_df: pd.DataFrame) -> List[int]:
    death_rows = labels_df[labels_df["name"].str.contains("Death", case=False, na=False)]
    exact = death_rows[death_rows["name"].str.strip() == "Death"]
    if len(exact) > 0:
        return exact["index"].astype(int).tolist()
    if len(death_rows) > 0:
        return death_rows["index"].astype(int).tolist()
    return [1269]


def get_patient_ids(data: np.ndarray, p2i: np.ndarray, n_patients: int) -> np.ndarray:
    starts = p2i[:n_patients, 0]
    return data[starts, 0].astype(int)


def collect_token_logits(
    model: Delphi,
    batch: Tuple[torch.Tensor, ...],
    token_indices: Sequence[int],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Run Delphi model and extract logits for specified token indices.
    Pass only (x, a) so the model runs in inference mode (no loss). Passing
    targets would trigger cross_entropy and fail when targets contain 1270
    (Death after +1 shift) with a checkpoint that has vocab_size=1270.
    """
    x, a, y, b = batch
    splits = [torch.split(tensor, batch_size) for tensor in [x, a]]
    logits = []
    with torch.no_grad():
        for mini_x, mini_a in zip(*splits):
            mini_x = mini_x.to(device)
            mini_a = mini_a.to(device)
            outputs = model(mini_x, mini_a)[0].detach().cpu().numpy()
            logits.append(outputs[:, :, token_indices])
    return np.vstack(logits)


def build_risk_scores(logits: np.ndarray) -> np.ndarray:
    """
    Aggregate per-patient logits to a scalar risk score.
    Strategy: max logit across sequence and death-token variants.
    """
    if logits.size == 0:
        return np.array([], dtype=np.float32)
    return logits.max(axis=(1, 2))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_delphi(args):
    device = args.device

    # Load model
    print(f"Loading Delphi model from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(device)

    # Load cohort + survival data for metrics
    cohort = data_access.load_cohort(Path(args.cohort_json))
    survival_df = data_access.load_survival_dataframe(Path(args.survival_csv))
    data_access.assert_dataset_matches(cohort, Path(args.survival_csv))

    # Load evaluation data (binary)
    data_dir = Path(args.data_path)
    split = args.split
    bin_path = data_dir / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"No binary data found at {bin_path}")

    print(f"Loading {split} data from {bin_path}...")
    data = np.fromfile(str(bin_path), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    data_p2i = get_p2i(data)

    n_patients = len(data_p2i)
    if args.max_patients > 0:
        n_patients = min(n_patients, args.max_patients)
    patient_eids = get_patient_ids(data, data_p2i, n_patients)
    print(f"Evaluating on {n_patients} patients ({split} split)...")

    d_batch = get_batch(
        range(n_patients),
        data,
        data_p2i,
        select="left",
        block_size=80,
        device=device,
        padding="random",
        no_event_token_rate=args.no_event_token_rate,
    )

    # Load labels + death tokens
    delphi_labels = pd.read_csv(DELPHI_LABELS)
    death_token_ids = parse_death_tokens(delphi_labels)
    print(f"Death token indices: {death_token_ids}")

    # Inference for death tokens
    death_logits = collect_token_logits(model, d_batch, death_token_ids, args.batch_size, device)
    risk_scores = build_risk_scores(death_logits)
    risk_map = dict(zip(patient_eids.tolist(), risk_scores.tolist()))

    # Align with survival targets
    split_eids = cohort[f"{split}_eids"]
    split_df, missing_surv = data_access.align_split_dataframe(survival_df, split_eids)
    if missing_surv:
        print(f"[evaluate_delphi] Warning: {len(missing_surv)} {split} eids missing from survival CSV.")

    aligned_risk = []
    durations = []
    events = []
    aligned_eids = []
    missing_preds = []

    for _, row in split_df.iterrows():
        eid = int(row["eid"])
        score = risk_map.get(eid)
        if score is None:
            missing_preds.append(eid)
            continue
        aligned_risk.append(score)
        durations.append(float(row["duration_days"]))
        events.append(int(row["event_flag"]))
        aligned_eids.append(eid)

    if not aligned_risk:
        raise RuntimeError("No overlapping Delphi predictions with survival cohort.")

    aligned_risk = np.asarray(aligned_risk, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.int32)

    train_df, _ = data_access.align_split_dataframe(survival_df, cohort["train_eids"])
    train_struct = metrics.to_structured(
        train_df["duration_days"].to_numpy(dtype=np.float64),
        train_df["event_flag"].to_numpy(dtype=np.int32),
    )
    eval_struct = metrics.to_structured(durations, events)

    horizons = args.horizons_days
    if not horizons:
        horizons = metrics.derive_time_horizons(train_df["duration_days"].to_numpy(dtype=np.float64))
    horizons = sorted({int(h) for h in horizons if h > 0})

    split_metrics = metrics.compute_metrics(
        train_struct,
        eval_struct,
        aligned_risk,
        horizons,
        survival_probs=None,
    )
    split_metrics["size"] = len(aligned_eids)
    split_metrics["event_rate"] = float(events.mean())

    if args.save_preds:
        survival_eval.save_predictions(
            Path(args.output_dir) / "predictions",
            "delphi",
            split,
            aligned_eids,
            aligned_risk,
            horizons,
            survival_probs=None,
        )

    # Optional legacy DeLong summary
    df_unpooled, df_pooled = evaluate_auc_pipeline(
        model=model,
        d100k=d_batch,
        output_path=str(Path(args.output_dir)),
        delphi_labels=delphi_labels,
        diseases_of_interest=death_token_ids,
        filter_min_total=0,
        disease_chunk_size=64,
        age_groups=np.arange(40, 80, 5),
        offset=args.offset,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        meta_info={"method": "delphi", "cohort": "shared", "split": split},
    )
    death_results = df_pooled[df_pooled["name"].str.contains("Death", case=False, na=False)]

    coverage = data_access.coverage_report(split_eids, aligned_eids)
    coverage["missing_predictions"] = missing_preds[:5]

    results = {
        "method": "delphi",
        "split": split,
        "risk_strategy": "max_logit_per_patient",
        "horizons": horizons,
        "splits": {split: split_metrics},
        "coverage": coverage,
        "death_token_ids": death_token_ids,
        "delong_summary": death_results.to_dict("records"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"delphi_{split}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Delphi on the shared cohort.")
    parser.add_argument("--ckpt-path", type=str,
                        default=str(PROJECT_ROOT / "Delphi" / "Delphi-2M-respiratory" / "ckpt.pt"))
    parser.add_argument("--data-path", type=str,
                        default=str(PROJECT_ROOT / "Delphi" / "data" / "ukb_respiratory_data"))
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offset", type=float, default=0.1)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--max-patients", type=int, default=-1,
                        help="Max patients to evaluate (-1 for all).")
    parser.add_argument("--n-bootstrap", type=int, default=1)
    parser.add_argument("--survival-csv", type=str, default=str(SURVIVAL_CSV))
    parser.add_argument("--cohort-json", type=str, default=str(COHORT_SPLIT))
    parser.add_argument("--horizons-days", type=int, nargs="*", default=None,
                        help="Optional explicit evaluation horizons in days.")
    parser.add_argument("--save-preds", action="store_true",
                        help="If set, save risk predictions to output/predictions.")
    args = parser.parse_args()

    evaluate_delphi(args)


if __name__ == "__main__":
    main()
