#!/usr/bin/env python3
"""绘制图7：两步核岭在四类冷启动组合中的综合评分。"""

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
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import (  # noqa: E402
    PALETTE,
    apply_publication_style,
    audit_design,
    audit_layout,
)


MODEL_ORDER = ["stage4_hybrid_base", "two_step_kernel_ridge"]
MODEL_LABELS = ["混合岭基础分", "两步核岭"]
MODEL_COLORS = [PALETTE["known"], PALETTE["primary"]]
IMPROVEMENT_LABEL = "综合评分提升"
IMPROVEMENT_COLOR = PALETTE["secondary"]
PANEL_CONFIG = [
    ("旧基因", ["old_cell_old_gene", "new_cell_old_gene"]),
    ("新基因", ["old_cell_new_gene", "new_cell_new_gene"]),
]
CELL_LABELS = ["旧细胞", "新细胞"]


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


def load_scores() -> pd.DataFrame:
    """读取正式外层评分，并严格保留图表契约中的八个数值。"""

    source = RESULTS_DIR / "stage7_official_metrics.csv"
    metrics = pd.read_csv(source)
    required_columns = {"model", "subset", "final_score"}
    if not required_columns.issubset(metrics.columns):
        raise ValueError(f"阶段7评分表缺少字段：{sorted(required_columns - set(metrics.columns))}")

    required_subsets = [subset for _, subsets in PANEL_CONFIG for subset in subsets]
    selected = metrics[
        metrics["model"].isin(MODEL_ORDER)
        & metrics["subset"].isin(required_subsets)
    ][["model", "subset", "final_score"]].copy()

    if len(selected) != 8:
        raise ValueError(f"图7应恰好读取8行正式分组评分，当前为{len(selected)}行")
    if selected.duplicated(["model", "subset"]).any():
        raise ValueError("图7的模型—分组组合存在重复")
    expected_pairs = {(model, subset) for model in MODEL_ORDER for subset in required_subsets}
    actual_pairs = set(selected[["model", "subset"]].itertuples(index=False, name=None))
    if actual_pairs != expected_pairs:
        raise ValueError("图7的模型或冷启动分组与契约不一致")
    if not np.isfinite(selected["final_score"]).all():
        raise ValueError("图7综合评分中存在缺失或无穷值")
    if not selected["final_score"].between(0, 100).all():
        raise ValueError("图7综合评分超出0—100的预期范围")
    return selected


def panel_values(scores: pd.DataFrame, subsets: list[str]) -> np.ndarray:
    """按细胞新旧和模型顺序整理一个面板的二乘二数值。"""

    values = np.empty((len(subsets), len(MODEL_ORDER)), dtype=float)
    for cell_index, subset in enumerate(subsets):
        for model_index, model in enumerate(MODEL_ORDER):
            row = scores[(scores["subset"] == subset) & (scores["model"] == model)]
            values[cell_index, model_index] = float(row["final_score"].iloc[0])
    return values


def draw_figure() -> tuple[plt.Figure, pd.DataFrame]:
    """按照绝对成绩与改善量分层的契约绘制图7。"""

    apply_publication_style(language="zh", width="double")
    mpl.rcParams["figure.constrained_layout.use"] = False
    zh_font, en_font, en_bold_font = configure_fonts()
    scores = load_scores()

    fig = plt.figure(figsize=(7.2, 4.15), layout=None)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.1, 1.05],
        hspace=0.34,
        wspace=0.12,
    )
    absolute_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    improvement_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    absolute_axes[1].sharey(absolute_axes[0])
    improvement_axes[1].sharey(improvement_axes[0])
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.14, top=0.82)

    group_centers = np.array([0.0, 1.0])
    bar_width = 0.30
    offsets = np.array([-0.18, 0.18])

    for panel_index, (axis, improvement_axis, (panel_name, subsets)) in enumerate(
        zip(absolute_axes, improvement_axes, PANEL_CONFIG)
    ):
        values = panel_values(scores, subsets)
        for model_index, (label, color) in enumerate(zip(MODEL_LABELS, MODEL_COLORS)):
            bars = axis.bar(
                group_centers + offsets[model_index],
                values[:, model_index],
                width=bar_width,
                color=color,
                edgecolor="none",
                linewidth=0,
                label=label,
                zorder=2,
            )
            for bar, value in zip(bars, values[:, model_index]):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.15,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontproperties=en_font,
                    fontsize=7.3,
                    color=PALETTE["dark"],
                )

        axis.set_xlim(-0.55, 1.55)
        axis.set_ylim(0, 85)
        axis.set_xticks(group_centers)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
        axis.set_yticks([0, 20, 40, 60, 80])
        axis.tick_params(axis="y", length=3, width=0.6, pad=3)
        for label in axis.get_yticklabels():
            label.set_fontproperties(en_font)
            label.set_fontsize(7.0)

        axis.set_title(panel_name, loc="left", fontproperties=zh_font, fontsize=8.4, pad=8)
        axis.text(
            -0.10,
            1.055,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontproperties=en_bold_font,
            fontsize=8.4,
            color=PALETTE["dark"],
            clip_on=False,
        )
        axis.spines["bottom"].set_color(PALETTE["dark"])
        axis.spines["bottom"].set_linewidth(0.7)
        axis.spines["left"].set_color(PALETTE["dark"])
        axis.spines["left"].set_linewidth(0.7)

        improvements = values[:, 1] - values[:, 0]
        improvement_bars = improvement_axis.bar(
            group_centers,
            improvements,
            width=0.38,
            color=IMPROVEMENT_COLOR,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        for bar, value in zip(improvement_bars, improvements):
            improvement_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.065,
                f"+{value:.2f}",
                ha="center",
                va="bottom",
                fontproperties=en_font,
                fontsize=7.1,
                color=PALETTE["dark"],
            )
        improvement_axis.set_xlim(-0.55, 1.55)
        improvement_axis.set_ylim(0, 1.7)
        improvement_axis.set_xticks(group_centers)
        improvement_axis.set_xticklabels(CELL_LABELS, fontproperties=zh_font, fontsize=7.6)
        improvement_axis.tick_params(axis="x", length=0, pad=5)
        improvement_axis.set_yticks([0, 0.5, 1.0, 1.5])
        improvement_axis.tick_params(axis="y", length=3, width=0.6, pad=3)
        for label in improvement_axis.get_yticklabels():
            label.set_fontproperties(en_font)
            label.set_fontsize(6.8)
        improvement_axis.spines["bottom"].set_color(PALETTE["dark"])
        improvement_axis.spines["bottom"].set_linewidth(0.7)
        improvement_axis.spines["left"].set_color(PALETTE["dark"])
        improvement_axis.spines["left"].set_linewidth(0.7)

    absolute_axes[0].set_ylabel("综合评分", fontproperties=zh_font, fontsize=8.0, labelpad=7)
    improvement_axes[0].set_ylabel("提升量", fontproperties=zh_font, fontsize=8.0, labelpad=7)
    absolute_axes[1].spines["left"].set_visible(False)
    absolute_axes[1].tick_params(axis="y", left=False, labelleft=False)
    improvement_axes[1].spines["left"].set_visible(False)
    improvement_axes[1].tick_params(axis="y", left=False, labelleft=False)

    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in zip(MODEL_LABELS, MODEL_COLORS)
    ]
    handles.append(
        Patch(facecolor=IMPROVEMENT_COLOR, edgecolor="none", label=IMPROVEMENT_LABEL)
    )
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.955),
        ncol=3,
        frameon=False,
        handlelength=1.35,
        handleheight=0.78,
        columnspacing=1.5,
        handletextpad=0.55,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_fontproperties(zh_font)
        text.set_fontsize(7.5)

    layout_issues = audit_layout(fig)
    design_issues = audit_design(fig)
    if layout_issues or design_issues:
        raise ValueError("图7预检未通过：" + "；".join(layout_issues + design_issues))
    return fig, scores


def export_preview(fig: plt.Figure) -> Path:
    """只导出一张供交互审阅的PNG。"""

    output = FIGURES_DIR / "result_q2_cold_start_scores_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    return output


def export_final(fig: plt.Figure) -> list[Path]:
    """审阅通过后导出论文使用的完整格式。"""

    stem = FIGURES_DIR / "result_q2_cold_start_scores"
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    tiff_path = stem.with_suffix(".tiff")
    grayscale_path = FIGURES_DIR / "_qa" / f"{stem.name}_grayscale.png"
    grayscale_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    fig.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(png_path) as color_image:
        color_image.convert("RGB").convert("L").save(grayscale_path)
    return [
        png_path,
        svg_path,
        grayscale_path,
        pdf_path,
        tiff_path,
    ]


def improvement_summary(scores: pd.DataFrame) -> str:
    """输出四类组合的精确改善量，供核对但不写进图面。"""

    changes = []
    for _, subsets in PANEL_CONFIG:
        for subset in subsets:
            baseline = float(
                scores[(scores["model"] == MODEL_ORDER[0]) & (scores["subset"] == subset)][
                    "final_score"
                ].iloc[0]
            )
            winner = float(
                scores[(scores["model"] == MODEL_ORDER[1]) & (scores["subset"] == subset)][
                    "final_score"
                ].iloc[0]
            )
            changes.append(f"{subset}={winner - baseline:+.6f}")
    return "；".join(changes)


def main() -> None:
    """执行选定导出模式并输出数据核对摘要。"""

    args = parse_args()
    figure, scores = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    print("四类正式外层综合评分已核对，共8个模型—分组值。")
    print("精确改善量：" + improvement_summary(scores))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
