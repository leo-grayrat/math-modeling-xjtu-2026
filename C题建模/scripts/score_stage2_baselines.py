"""Score MATLAB stage-2 baselines with the competition's official functions."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
DATA_ROOT = REPO_ROOT / "2026年校赛题目" / "C" / "数据文件"
SCORING_DIR = PROJECT_DIR / "results" / "stage2_scoring"
OUTPUT_PATH = PROJECT_DIR / "results" / "stage2_baseline_metrics.csv"
MODEL_NAMES = (
    "global_mean",
    "cell_mean",
    "gene_mean",
    "cell_gene_additive",
)
SUBSETS = (
    "all",
    "old_cell_old_gene",
    "old_cell_new_gene",
    "new_cell_old_gene",
    "new_cell_new_gene",
)


def load_official_metric():
    metric_path = DATA_ROOT / "calculate_metric.py"
    spec = importlib.util.spec_from_file_location("official_metric", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official metric: {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_row(metric, frame: pd.DataFrame, model: str, subset: str) -> dict[str, object]:
    ranking = metric.metric_by_cell(frame)
    rmse = float(np.sqrt(np.mean(np.square(frame["prediction"] - frame["truth"]))))
    sigma_y = float(frame["truth"].std(ddof=0))
    nrmse = rmse / (sigma_y + metric.EPS)
    rmse_score = 1.0 / (1.0 + nrmse)
    final_score = 100.0 * (
        0.30 * ranking["spearman_score"]
        + 0.30 * ranking["ndcg_score"]
        + 0.25 * ranking["precision_score"]
        + 0.15 * rmse_score
    )

    return {
        "model": model,
        "subset": subset,
        "final_score": final_score,
        **ranking,
        "rmse_score": rmse_score,
        "rmse": rmse,
        "nrmse": nrmse,
        "rows": len(frame),
        "cell_lines": frame["cell_line_id"].nunique(),
    }


def main() -> None:
    metric = load_official_metric()
    answer = metric.read_answer(SCORING_DIR / "stage2_validation_answer.csv")
    groups = pd.read_csv(SCORING_DIR / "stage2_validation_groups.csv")
    if not groups[metric.KEYS].reset_index(drop=True).equals(
        answer[metric.KEYS].reset_index(drop=True)
    ):
        raise ValueError("Validation group rows do not match the answer rows.")

    rows: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        submission = metric.read_submission(SCORING_DIR / f"prediction_{model}.csv")
        metric.validate_pairs(submission, answer)
        merged = answer.merge(submission, on=metric.KEYS, how="left", validate="one_to_one")

        for subset in SUBSETS:
            if subset == "all":
                frame = merged
            else:
                frame = merged.loc[groups["group"].eq(subset).to_numpy()].copy()
            if frame.empty:
                raise ValueError(f"Empty scoring subset: {subset}")
            rows.append(calculate_row(metric, frame, model, subset))

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")

    overall = result[result["subset"].eq("all")][
        ["model", "final_score", "rho_macro", "ndcg_score", "precision_score", "rmse"]
    ]
    print(overall.to_string(index=False))
    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
