#!/usr/bin/env python3
"""用官方评分函数整理图5所需的阶段4基础分模型指标。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DATA_ROOT = REPO_ROOT / "2026年校赛题目" / "C" / "数据文件"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "stage2_scoring"
PREDICTION_DIR = RESULTS_DIR / "stage4_scoring"
OUTPUT_PATH = RESULTS_DIR / "figure5_gene_base_model_comparison.csv"

MODEL_ORDER = (
    "gene_mean",
    "metadata_ridge",
    "metadata_lasso",
    "hybrid_ridge",
    "hybrid_lasso",
)
SUBSET_ORDER = (
    "all",
    "old_cell_new_gene",
    "new_cell_new_gene",
)


def load_official_metric():
    """加载赛题提供的官方评分模块。"""

    metric_path = DATA_ROOT / "calculate_metric.py"
    spec = importlib.util.spec_from_file_location("official_metric", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载官方评分脚本：{metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_metrics(metric, frame: pd.DataFrame) -> dict[str, float]:
    """按官方权重计算综合评分和RMSE。"""

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
    return {"final_score": final_score, "rmse": rmse}


def build_metrics() -> pd.DataFrame:
    """评分五份冻结预测，并保留图5需要的三个分组。"""

    metric = load_official_metric()
    answer = metric.read_answer(SPLIT_DIR / "stage2_validation_answer.csv")
    groups = pd.read_csv(SPLIT_DIR / "stage2_validation_groups.csv")
    if len(answer) != 265065 or len(groups) != 265065:
        raise ValueError("阶段4固定外层验证规模不再是265065行")
    if not groups[metric.KEYS].reset_index(drop=True).equals(
        answer[metric.KEYS].reset_index(drop=True)
    ):
        raise ValueError("固定外层分组与答案行未能一一对应")

    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        submission = metric.read_submission(PREDICTION_DIR / f"prediction_{model}.csv")
        metric.validate_pairs(submission, answer)
        merged = answer.merge(
            submission,
            on=metric.KEYS,
            how="left",
            validate="one_to_one",
        )
        for subset in SUBSET_ORDER:
            frame = (
                merged
                if subset == "all"
                else merged.loc[groups["group"].eq(subset).to_numpy()].copy()
            )
            if frame.empty:
                raise ValueError(f"图5评分分组为空：{subset}")
            rows.append(
                {
                    "model": model,
                    "subset": subset,
                    **calculate_metrics(metric, frame),
                    "rows": len(frame),
                }
            )

    result = pd.DataFrame(rows)
    gene_mean_rmse = (
        result[result["model"].eq("gene_mean")]
        .set_index("subset")["rmse"]
        .to_dict()
    )
    result["rmse_reduction_vs_gene_mean"] = [
        gene_mean_rmse[row.subset] - row.rmse
        if row.subset in {"old_cell_new_gene", "new_cell_new_gene"}
        else np.nan
        for row in result.itertuples()
    ]
    if len(result) != len(MODEL_ORDER) * len(SUBSET_ORDER):
        raise ValueError("图5派生表应恰好包含15行")
    return result


def main() -> None:
    """写出可追溯的小型绘图数据表。"""

    result = build_metrics()
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")
    overall = result[result["subset"].eq("all")][["model", "final_score", "rmse"]]
    new_gene = result[
        result["model"].isin(["hybrid_ridge", "hybrid_lasso"])
        & result["subset"].isin(["old_cell_new_gene", "new_cell_new_gene"])
    ][["model", "subset", "rmse_reduction_vs_gene_mean"]]
    print("五种基础分方案的正式外层综合评分：")
    print(overall.to_string(index=False))
    print("\n两个新基因组相对基因均值的RMSE降低量：")
    print(new_gene.to_string(index=False))
    print(f"\n绘图数据已写入：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
