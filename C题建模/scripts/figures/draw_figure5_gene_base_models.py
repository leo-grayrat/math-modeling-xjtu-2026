#!/usr/bin/env python3
"""绘制图5：五种基因基础分方案的综合表现与新基因误差改善。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib as mpl

# 服务器或终端环境中不打开交互窗口。
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_PATH = RESULTS_DIR / "figure5_gene_base_model_comparison.csv"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import (  # noqa: E402
    CONFIDENCE_COLORS,
    PALETTE,
    apply_publication_style,
    audit_design,
    audit_layout,
)


MODEL_ORDER = [
    "hybrid_ridge",
    "hybrid_lasso",
    "gene_mean",
    "metadata_ridge",
    "metadata_lasso",
]
MODEL_LABELS = {
    "hybrid_ridge": "混合岭",
    "hybrid_lasso": "混合 LASSO",
    "gene_mean": "基因均值",
    "metadata_ridge": "纯元数据岭",
    "metadata_lasso": "纯元数据 LASSO",
}
MODEL_COLORS = {
    "hybrid_ridge": CONFIDENCE_COLORS["high"],
    "hybrid_lasso": CONFIDENCE_COLORS["high"],
    "gene_mean": CONFIDENCE_COLORS["medium"],
    "metadata_ridge": CONFIDENCE_COLORS["low"],
    "metadata_lasso": CONFIDENCE_COLORS["low"],
}
# 两个改善量都属于正向结果，仅用同一绿色的深浅区分算法。
IMPROVEMENT_COLORS = {
    "hybrid_ridge": "#3FAE7C",
    "hybrid_lasso": "#A8E5CA",
}
NEW_GENE_SUBSETS = ["old_cell_new_gene", "new_cell_new_gene"]
NEW_GENE_LABELS = ["旧细胞", "新细胞"]


def parse_args() -> argparse.Namespace:
    """解析预览和正式导出模式。"""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="只生成一张PNG预览")
    mode.add_argument("--final", action="store_true", help="生成论文使用的完整格式")
    return parser.parse_args()


def find_font(relative_path: str) -> Path:
    """从常见用户级目录定位项目规定字体。"""

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / relative_path,
        Path.home() / "AppData" / "Local" / relative_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到绘图所需字体：{relative_path}")


def configure_fonts() -> tuple[FontProperties, FontProperties, FontProperties]:
    """注册中文、英文常规和英文粗体字体。"""

    zh_path = find_font("Microsoft/Windows/Fonts/FandolSong-Regular.otf")
    en_path = find_font(
        "Programs/MiKTeX/fonts/opentype/public/tex-gyre/"
        "texgyretermes-regular.otf"
    )
    en_bold_path = find_font(
        "Programs/MiKTeX/fonts/opentype/public/tex-gyre/"
        "texgyretermes-bold.otf"
    )
    for path in (zh_path, en_path, en_bold_path):
        fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "TeX Gyre Termes",
                "FandolSong",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "cm",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return (
        FontProperties(fname=zh_path),
        FontProperties(fname=en_path),
        FontProperties(fname=en_bold_path),
    )


def load_data() -> pd.DataFrame:
    """读取并核对评分派生表。"""

    data = pd.read_csv(DATA_PATH)
    required = {
        "model",
        "subset",
        "final_score",
        "rmse",
        "rmse_reduction_vs_gene_mean",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"图5派生表缺少字段：{sorted(required - set(data.columns))}")
    expected_subsets = {"all", *NEW_GENE_SUBSETS}
    if len(data) != 15 or set(data["model"]) != set(MODEL_ORDER):
        raise ValueError("图5派生表必须包含五种模型各三个分组")
    if set(data["subset"]) != expected_subsets:
        raise ValueError("图5派生表分组与契约不一致")
    if data.duplicated(["model", "subset"]).any():
        raise ValueError("图5派生表存在重复模型—分组组合")
    if not np.isfinite(data[["final_score", "rmse"]]).all().all():
        raise ValueError("图5派生表中存在缺失或无穷数值")
    return data


def pick(data: pd.DataFrame, model: str, subset: str, column: str) -> float:
    """读取唯一的模型—分组指标。"""

    row = data[(data["model"] == model) & (data["subset"] == subset)]
    if len(row) != 1:
        raise ValueError(f"无法唯一定位图5指标：{model} / {subset}")
    return float(row.iloc[0][column])


def draw_figure() -> tuple[plt.Figure, pd.DataFrame]:
    """按照模型比较契约绘制图5预览。"""

    apply_publication_style(language="zh", width="double")
    mpl.rcParams["figure.constrained_layout.use"] = False
    zh_font, en_font, en_bold_font = configure_fonts()
    data = load_data()

    fig = plt.figure(figsize=(7.2, 3.75), layout=None)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.30, 0.90], wspace=0.31)
    score_axis = fig.add_subplot(grid[0, 0])
    rmse_axis = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.18, top=0.80)

    score_values = np.array([
        pick(data, model, "all", "final_score") for model in MODEL_ORDER
    ])
    y_positions = np.arange(len(MODEL_ORDER))
    score_bars = score_axis.barh(
        y_positions,
        score_values,
        height=0.58,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    for bar, value in zip(score_bars, score_values):
        score_axis.text(
            value + 0.75,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontproperties=en_font,
            fontsize=7.2,
            color=PALETTE["dark"],
        )
    score_axis.set_yticks(y_positions)
    score_axis.set_yticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        fontproperties=zh_font,
        fontsize=7.5,
    )
    score_axis.tick_params(axis="y", length=0, pad=7)
    score_axis.set_xlim(0, 60)
    score_axis.set_xticks([0, 20, 40, 60])
    score_axis.set_xlabel("综合评分", fontproperties=zh_font, fontsize=7.8, labelpad=5)
    score_axis.tick_params(axis="x", length=3, width=0.6, pad=3)
    for label in score_axis.get_xticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(7.0)
    score_axis.set_title("五种基础分方案", loc="left", fontproperties=zh_font, fontsize=8.4, pad=9)
    score_axis.text(
        -0.08,
        1.055,
        "a",
        transform=score_axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=en_bold_font,
        fontsize=8.4,
        color=PALETTE["dark"],
        clip_on=False,
    )

    x_positions = np.array([0.0, 1.0])
    width = 0.29
    for model_index, model in enumerate(["hybrid_ridge", "hybrid_lasso"]):
        reduction_values = np.array([
            pick(data, model, subset, "rmse_reduction_vs_gene_mean")
            for subset in NEW_GENE_SUBSETS
        ])
        bars = rmse_axis.bar(
            x_positions + (model_index - 0.5) * width,
            reduction_values,
            width=width,
            color=IMPROVEMENT_COLORS[model],
            edgecolor="none",
            linewidth=0,
            label=MODEL_LABELS[model],
            zorder=2,
        )
        for bar, value in zip(bars, reduction_values):
            rmse_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.00045,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontproperties=en_font,
                fontsize=6.8,
                color=PALETTE["dark"],
            )
    rmse_axis.set_xlim(-0.55, 1.55)
    rmse_axis.set_ylim(0, 0.020)
    rmse_axis.set_xticks(x_positions)
    rmse_axis.set_xticklabels(NEW_GENE_LABELS, fontproperties=zh_font, fontsize=7.5)
    rmse_axis.tick_params(axis="x", length=0, pad=6)
    rmse_axis.set_yticks([0, 0.005, 0.010, 0.015, 0.020])
    rmse_axis.set_ylabel("RMSE降低量", fontproperties=zh_font, fontsize=7.8, labelpad=6)
    rmse_axis.tick_params(axis="y", length=3, width=0.6, pad=3)
    for label in rmse_axis.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(6.8)
    rmse_axis.set_title("新基因组误差改善", loc="left", fontproperties=zh_font, fontsize=8.4, pad=9)
    rmse_axis.text(
        -0.11,
        1.055,
        "b",
        transform=rmse_axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=en_bold_font,
        fontsize=8.4,
        color=PALETTE["dark"],
        clip_on=False,
    )
    rmse_axis.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.045),
        ncol=2,
        handlelength=1.15,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0,
        prop=zh_font,
    )
    for text in rmse_axis.get_legend().get_texts():
        text.set_fontsize(6.5)

    for axis in (score_axis, rmse_axis):
        axis.spines["bottom"].set_color(PALETTE["dark"])
        axis.spines["bottom"].set_linewidth(0.7)
        axis.spines["left"].set_color(PALETTE["dark"])
        axis.spines["left"].set_linewidth(0.7)
    layout_issues = audit_layout(fig)
    design_issues = audit_design(fig)
    if layout_issues or design_issues:
        raise ValueError("图5预检未通过：" + "；".join(layout_issues + design_issues))
    return fig, data


def export_preview(fig: plt.Figure) -> Path:
    """只导出一张供交互审阅的PNG。"""

    output = FIGURES_DIR / "result_q2_gene_base_model_comparison_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    return output


def export_final(fig: plt.Figure) -> list[Path]:
    """审阅通过后导出论文使用的完整格式。"""

    stem = FIGURES_DIR / "result_q2_gene_base_model_comparison"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    tiff_path = stem.with_suffix(".tiff")
    grayscale_path = FIGURES_DIR / "_qa" / f"{stem.name}_grayscale.png"
    grayscale_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(pdf_path)
    fig.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image

    with Image.open(png_path) as color_image:
        color_image.convert("RGB").convert("L").save(grayscale_path)
    return [png_path, svg_path, pdf_path, tiff_path, grayscale_path]


def main() -> None:
    """执行选定导出模式。"""

    args = parse_args()
    figure, _ = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
