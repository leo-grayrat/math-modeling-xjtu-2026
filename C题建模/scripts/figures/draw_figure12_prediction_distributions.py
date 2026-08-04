#!/usr/bin/env python3
"""绘制图12：四类新旧对象组合的正式预测分布。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib as mpl

# 终端运行时不打开交互窗口。
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
TRAIN_PATH = (
    PROJECT_ROOT.parent
    / "2026年校赛题目"
    / "C"
    / "数据文件"
    / "labels"
    / "gene_dependency.csv"
)
PREDICTION_PATH = RESULTS_DIR / "final_submission" / "stage10_submission_gene.csv"
GROUP_SUMMARY_PATH = RESULTS_DIR / "stage1_official_test_groups.csv"
STAGE10_SUMMARY_PATH = RESULTS_DIR / "stage10_summary.csv"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import PALETTE, apply_publication_style, audit_design, audit_layout  # noqa: E402


GROUP_ORDER = [
    "old_cell_old_gene",
    "new_cell_old_gene",
    "old_cell_new_gene",
    "new_cell_new_gene",
]
GROUP_LABELS = {
    "old_cell_old_gene": "旧细胞—旧基因",
    "new_cell_old_gene": "新细胞—旧基因",
    "old_cell_new_gene": "旧细胞—新基因",
    "new_cell_new_gene": "新细胞—新基因",
}
GROUP_COLORS = {
    "old_cell_old_gene": PALETTE["known"],
    "new_cell_old_gene": "#BBC4CC",
    "old_cell_new_gene": PALETTE["new"],
    "new_cell_new_gene": "#FFE7B3",
}


def parse_args() -> argparse.Namespace:
    """解析预览和正式导出模式。"""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="只生成一张PNG预览")
    mode.add_argument("--final", action="store_true", help="生成论文使用的完整格式")
    return parser.parse_args()


def find_font(relative_path: str) -> Path:
    """从常见用户目录定位项目规定字体。"""

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / relative_path,
        Path.home() / "AppData" / "Local" / relative_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到绘图所需字体：{relative_path}")


def configure_fonts() -> tuple[FontProperties, FontProperties]:
    """注册中文和英文常规字体。"""

    zh_path = find_font("Microsoft/Windows/Fonts/FandolSong-Regular.otf")
    en_path = find_font(
        "Programs/MiKTeX/fonts/opentype/public/tex-gyre/"
        "texgyretermes-regular.otf"
    )
    for path in (zh_path, en_path):
        fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Termes", "FandolSong", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return FontProperties(fname=zh_path), FontProperties(fname=en_path)


def load_grouped_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """用训练集中出现过的细胞和基因标记正式预测的四类组合。"""

    training = pd.read_csv(
        TRAIN_PATH,
        usecols=["cell_line_id", "perturbation_gene"],
        dtype={"cell_line_id": "string", "perturbation_gene": "string"},
    )
    predictions = pd.read_csv(
        PREDICTION_PATH,
        dtype={"cell_line_id": "string", "perturbation_gene": "string"},
    )
    required_prediction_columns = {"cell_line_id", "perturbation_gene", "label"}
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError("正式预测文件缺少细胞、基因或预测值字段")
    if len(predictions) != 373990:
        raise ValueError(f"正式预测应有373990行，当前为{len(predictions)}行")
    if predictions.duplicated(["cell_line_id", "perturbation_gene"]).any():
        raise ValueError("正式预测中存在重复的细胞—基因组合")
    if not np.isfinite(predictions["label"]).all():
        raise ValueError("正式预测中存在缺失或无穷值")

    if training[["cell_line_id", "perturbation_gene"]].isna().any().any():
        raise ValueError("训练标签的细胞或基因字段存在缺失，不能静默排除")
    old_cells = set(training["cell_line_id"].astype(str))
    old_genes = set(training["perturbation_gene"].astype(str))
    cell_seen = predictions["cell_line_id"].astype(str).isin(old_cells)
    gene_seen = predictions["perturbation_gene"].astype(str).isin(old_genes)
    predictions["group"] = np.select(
        [
            cell_seen & gene_seen,
            ~cell_seen & gene_seen,
            cell_seen & ~gene_seen,
            ~cell_seen & ~gene_seen,
        ],
        GROUP_ORDER,
        default="unclassified",
    )
    if (predictions["group"] == "unclassified").any():
        raise ValueError("存在无法归入四类新旧对象组合的预测")

    expected = pd.read_csv(GROUP_SUMMARY_PATH).set_index("group")["row_count"]
    actual = predictions["group"].value_counts().reindex(GROUP_ORDER)
    if not actual.equals(expected.reindex(GROUP_ORDER)):
        raise ValueError("按训练对象集合重算的四类数量与阶段1汇总不一致")

    stage10 = pd.read_csv(STAGE10_SUMMARY_PATH)
    if len(stage10) != 1:
        raise ValueError("阶段10汇总必须恰好包含一行")
    row = stage10.iloc[0]
    if int(row["prediction_rows"]) != len(predictions):
        raise ValueError("正式预测行数与阶段10汇总不一致")
    if not np.isclose(float(row["prediction_min"]), predictions["label"].min()):
        raise ValueError("正式预测最小值与阶段10汇总不一致")
    if not np.isclose(float(row["prediction_max"]), predictions["label"].max()):
        raise ValueError("正式预测最大值与阶段10汇总不一致")
    return predictions, actual


def density_curves(
    predictions: pd.DataFrame,
    bins: int = 520,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """用全部预测计算统一网格上的平滑直方密度，不进行抽样。"""

    lower = float(predictions["label"].min())
    upper = float(predictions["label"].max())
    edges = np.linspace(lower, upper, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    curves: dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        values = predictions.loc[predictions["group"].eq(group), "label"].to_numpy(dtype=float)
        density, _ = np.histogram(values, bins=edges, density=True)
        curves[group] = gaussian_filter1d(density.astype(float), sigma=2.4, mode="nearest")
    return centers, curves


def draw_figure() -> tuple[plt.Figure, pd.Series]:
    """按照四行共享横轴的山脊分布契约绘制图12。"""

    apply_publication_style(language="zh", width="double")
    mpl.rcParams["figure.constrained_layout.use"] = False
    zh_font, en_font = configure_fonts()
    predictions, counts = load_grouped_predictions()
    x_values, curves = density_curves(predictions)
    display_upper = float(predictions["label"].quantile(0.995))

    figure, axis = plt.subplots(figsize=(7.2, 3.25), layout=None)
    figure.subplots_adjust(left=0.255, right=0.975, bottom=0.20, top=0.95)
    baselines = np.arange(len(GROUP_ORDER))[::-1].astype(float)
    global_peak = max(float(curve.max()) for curve in curves.values())
    ridge_height = 0.74

    for baseline, group in zip(baselines, GROUP_ORDER):
        scaled_density = curves[group] / global_peak * ridge_height
        color = GROUP_COLORS[group]
        axis.fill_between(
            x_values,
            baseline,
            baseline + scaled_density,
            color=color,
            alpha=0.92,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            x_values,
            baseline + scaled_density,
            color=color,
            linewidth=1.0,
            zorder=3,
        )
        axis.hlines(
            baseline,
            x_values[0],
            x_values[-1],
            color="#D9DEDA",
            linewidth=0.55,
            zorder=1,
        )

    axis.set_yticks(baselines + 0.15)
    labels = [f"{GROUP_LABELS[group]}  n={int(counts[group]):,}" for group in GROUP_ORDER]
    axis.set_yticklabels(labels, fontproperties=zh_font, fontsize=7.3)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.set_xlim(float(predictions["label"].min()), display_upper)
    axis.set_xticks([-0.4, 0, 0.4, 0.8, 1.2, 1.6])
    axis.set_xlabel("正式预测值", fontproperties=zh_font, fontsize=7.8, labelpad=6)
    axis.tick_params(axis="x", length=3, width=0.6, pad=4)
    for label in axis.get_xticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(7.0)
    axis.set_ylim(-0.12, 3.86)
    axis.grid(axis="x", color="#E8ECE9", linewidth=0.55, zorder=0)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(PALETTE["dark"])
    axis.spines["bottom"].set_linewidth(0.7)

    layout_issues = audit_layout(figure)
    design_issues = audit_design(figure)
    if layout_issues or design_issues:
        raise ValueError("图12预检未通过：" + "；".join(layout_issues + design_issues))
    return figure, counts


def export_preview(figure: plt.Figure) -> Path:
    """只导出一张供交互审阅的PNG。"""

    output = FIGURES_DIR / "result_q2_prediction_distributions_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    return output


def export_final(figure: plt.Figure) -> list[Path]:
    """用户确认后可导出论文使用的完整格式。"""

    stem = FIGURES_DIR / "result_q2_prediction_distributions"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    tiff_path = stem.with_suffix(".tiff")
    grayscale_path = FIGURES_DIR / "_qa" / f"{stem.name}_grayscale.png"
    grayscale_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300)
    figure.savefig(svg_path)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(pdf_path)
    figure.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(png_path) as color_image:
        color_image.convert("RGB").convert("L").save(grayscale_path)
    return [png_path, svg_path, pdf_path, tiff_path, grayscale_path]


def main() -> None:
    """执行选定导出模式并打印四类预测数量。"""

    args = parse_args()
    figure, counts = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    print("图12使用全部373990条正式预测计算密度，横轴显示99.5%的预测值范围。")
    print("四类数量：" + "；".join(f"{group}={int(counts[group])}" for group in GROUP_ORDER))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
