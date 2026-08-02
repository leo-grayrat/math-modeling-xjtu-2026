"""使用赛题官方函数评分阶段7唯一的外层胜出路线。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
DATA_ROOT = REPO_ROOT / "2026年校赛题目" / "C" / "数据文件"
RESULTS_DIR = PROJECT_DIR / "results"
SCORING_DIR = RESULTS_DIR / "stage7_scoring"
OUTPUT_PATH = RESULTS_DIR / "stage7_official_metrics.csv"
SUBSETS = (
    "all",
    "old_cell_old_gene",
    "old_cell_new_gene",
    "new_cell_old_gene",
    "new_cell_new_gene",
)


def load_official_metric():
    """加载赛题附带的评分模块。"""
    metric_path = DATA_ROOT / "calculate_metric.py"
    spec = importlib.util.spec_from_file_location("official_metric", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载官方评分脚本：{metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_row(metric, frame: pd.DataFrame, model: str, subset: str) -> dict[str, object]:
    """按官方权重整理一行分组指标。"""
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
    """只读取 MATLAB 已冻结的外层胜出路线，不在这里调参。"""
    selection = pd.read_csv(RESULTS_DIR / "stage7_selected_model.csv")
    if not bool(selection.loc[0, "outerPredictionWritten"]):
        raise RuntimeError("内层没有产生可进入外层的路线，不应运行外层评分。")
    winner = str(selection.loc[0, "route"])
    model_names = ("stage4_hybrid_base", winner)

    metric = load_official_metric()
    answer = metric.read_answer(SCORING_DIR / "stage7_validation_answer.csv")
    groups = pd.read_csv(SCORING_DIR / "stage7_validation_groups.csv")
    if not groups[metric.KEYS].reset_index(drop=True).equals(
        answer[metric.KEYS].reset_index(drop=True)
    ):
        raise ValueError("外层分组行与答案行不一致。")

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        submission = metric.read_submission(
            SCORING_DIR / f"prediction_{model_name}.csv"
        )
        metric.validate_pairs(submission, answer)
        merged = answer.merge(
            submission,
            on=metric.KEYS,
            how="left",
            validate="one_to_one",
        )
        for subset in SUBSETS:
            frame = (
                merged
                if subset == "all"
                else merged.loc[groups["group"].eq(subset).to_numpy()].copy()
            )
            if frame.empty:
                raise ValueError(f"外层评分子集为空：{subset}")
            rows.append(calculate_row(metric, frame, model_name, subset))

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.10f")
    overall = result.loc[result["subset"].eq("all"), [
        "model",
        "final_score",
        "rho_macro",
        "ndcg_score",
        "precision_score",
        "rmse",
    ]]
    print(overall.to_string(index=False))
    print(f"官方指标已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
