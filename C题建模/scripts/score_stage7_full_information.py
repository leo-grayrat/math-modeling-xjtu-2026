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
GATE_OUTPUT_PATH = RESULTS_DIR / "stage7_gate_decision.csv"
MIN_SCORE_IMPROVEMENT = 0.20
MIN_IMPROVED_FOLDS = 2
MIN_IMPROVED_RANKING_METRICS = 2
MAX_RMSE_RELATIVE_WORSENING = 0.01
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


def build_gate_decision(
    result: pd.DataFrame,
    selection: pd.DataFrame,
    winner: str,
) -> pd.DataFrame:
    """把 Issue #18 的预设门槛整理成一行可复核结论。"""

    def pick(model: str, subset: str) -> pd.Series:
        row = result.loc[result["model"].eq(model) & result["subset"].eq(subset)]
        if len(row) != 1:
            raise ValueError(f"评分结果无法唯一定位：{model} / {subset}")
        return row.iloc[0]

    baseline = pick("stage4_hybrid_base", "all")
    winner_all = pick(winner, "all")
    old_cell_new_gene_base = pick("stage4_hybrid_base", "old_cell_new_gene")
    old_cell_new_gene_winner = pick(winner, "old_cell_new_gene")
    new_cell_new_gene_base = pick("stage4_hybrid_base", "new_cell_new_gene")
    new_cell_new_gene_winner = pick(winner, "new_cell_new_gene")

    ranking_metrics = ("rho_macro", "ndcg_score", "precision_score")
    improved_ranking_metrics = sum(
        float(winner_all[name]) > float(baseline[name])
        for name in ranking_metrics
    )
    score_improvement = float(winner_all["final_score"] - baseline["final_score"])
    old_cell_new_gene_improvement = float(
        old_cell_new_gene_winner["final_score"]
        - old_cell_new_gene_base["final_score"]
    )
    new_cell_new_gene_improvement = float(
        new_cell_new_gene_winner["final_score"]
        - new_cell_new_gene_base["final_score"]
    )
    rmse_relative_change = float(
        winner_all["rmse"] / baseline["rmse"] - 1.0
    )
    folds_improved = int(selection.loc[0, "folds_improved"])
    adopted = (
        folds_improved >= MIN_IMPROVED_FOLDS
        and score_improvement >= MIN_SCORE_IMPROVEMENT
        and old_cell_new_gene_improvement > 0
        and new_cell_new_gene_improvement > 0
        and improved_ranking_metrics >= MIN_IMPROVED_RANKING_METRICS
        and rmse_relative_change <= MAX_RMSE_RELATIVE_WORSENING
    )

    return pd.DataFrame([{
        "winner_route": winner,
        "lambda": float(selection.loc[0, "lambda"]),
        "inner_folds_improved": folds_improved,
        "inner_mean_improvement": float(selection.loc[0, "mean_improvement"]),
        "outer_baseline_score": float(baseline["final_score"]),
        "outer_winner_score": float(winner_all["final_score"]),
        "outer_score_improvement": score_improvement,
        "old_cell_new_gene_improvement": old_cell_new_gene_improvement,
        "new_cell_new_gene_improvement": new_cell_new_gene_improvement,
        "improved_ranking_metrics": improved_ranking_metrics,
        "rmse_relative_change": rmse_relative_change,
        "adopted": adopted,
    }])


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
    gate_decision = build_gate_decision(result, selection, winner)
    gate_decision.to_csv(GATE_OUTPUT_PATH, index=False, float_format="%.10f")
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
    print(gate_decision.to_string(index=False))
    print(f"阶段验收结论已写入 {GATE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
