#!/usr/bin/env python3
"""绘制图 2：数据结构与四类冷启动。"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import (  # noqa: E402
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """读取组合分组和对象覆盖数据，不在绘图阶段重新计算模型结果。"""

    groups = pd.read_csv(RESULTS_DIR / "stage1_official_test_groups.csv")
    summary = pd.read_csv(RESULTS_DIR / "stage1_dataset_summary.csv")
    summary = summary.set_index("metric")["value"]
    expected = {
        "old_cell_old_gene",
        "old_cell_new_gene",
        "new_cell_old_gene",
        "new_cell_new_gene",
    }
    if set(groups["group"]) != expected:
        raise ValueError("正式测试分组没有覆盖四类冷启动组合")
    if groups["row_count"].sum() != int(summary["test_rows"]):
        raise ValueError("四类组合行数与正式测试总行数不一致")
    return groups, summary


def draw() -> dict[str, str]:
    apply_publication_style(language="zh", width="report")
    # 把关键出版设置显式放在入口，便于静态预检和后续复现。
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["FandolSong", "Times New Roman", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    groups, summary = load_data()
    labels = {
        "old_cell_old_gene": (0, 0),
        "old_cell_new_gene": (0, 1),
        "new_cell_old_gene": (1, 0),
        "new_cell_new_gene": (1, 1),
    }
    groups["row"], groups["col"] = zip(*groups["group"].map(labels))
    total_rows = groups["row_count"].sum()
    cmap = LinearSegmentedColormap.from_list(
        "fresh_structure",
        ["#EAF3F1", PALETTE["secondary"], PALETTE["contrast"]],
    )
    norm = Normalize(vmin=groups["row_count"].min(), vmax=groups["row_count"].max())

    fig, axes = publication_subplots(
        1,
        2,
        width="report",
        aspect=0.55,
        width_ratios=[1.08, 1],
    )
    ax_matrix, ax_objects = axes

    # 面板 a：格子表达对象组合，圆的面积和明暗共同表达数量。
    for _, row in groups.iterrows():
        x, y = row["col"], 1 - row["row"]
        ax_matrix.add_patch(
            mpl.patches.Rectangle(
                (x - 0.48, y - 0.48),
                0.96,
                0.96,
                facecolor="#F8FAF9",
                edgecolor="#DCE8E4",
                linewidth=0.8,
                zorder=0,
            )
        )
        radius = 0.15 + 0.28 * (row["row_count"] / groups["row_count"].max()) ** 0.5
        ax_matrix.add_patch(
            mpl.patches.Circle(
                (x, y),
                radius=radius,
                facecolor=cmap(norm(row["row_count"])),
                edgecolor="white",
                linewidth=1.2,
                zorder=2,
            )
        )
        ax_matrix.text(
            x,
            y + 0.03,
            f"{int(row['row_count']):,}",
            ha="center",
            va="center",
            color=PALETTE["dark"],
            fontsize=8.2,
            fontweight="bold",
            zorder=3,
        )
        ax_matrix.text(
            x,
            y - 0.18,
            f"{row['row_count'] / total_rows:.1%}",
            ha="center",
            va="center",
            color=PALETTE["dark"],
            fontsize=7,
            zorder=3,
        )
    ax_matrix.set(
        title="对象组合",
        xticks=[0, 1],
        xticklabels=["基因已见", "基因未见"],
        yticks=[1, 0],
        yticklabels=["细胞已见", "细胞未见"],
        xlim=(-0.55, 1.55),
        ylim=(-0.55, 1.55),
    )
    ax_matrix.tick_params(axis="both", length=0, pad=5)
    ax_matrix.spines[:].set_visible(False)
    ax_matrix.set_aspect("equal")

    # 面板 b：对象覆盖，用颜色区分已见/新增，用长度表达数量。
    train_cells = int(summary["train_unique_cells"])
    new_cells = int(summary["new_test_cells"])
    train_genes = int(summary["train_unique_genes"])
    new_genes = int(summary["new_test_genes"])
    object_rows = [
        ("细胞", train_cells, new_cells),
        ("基因", train_genes, new_genes),
    ]
    y_positions = [1, 0]
    max_total = max(train_cells + new_cells, train_genes + new_genes)
    for y, (name, known, new) in zip(y_positions, object_rows):
        ax_objects.barh(
            y,
            known,
            height=0.42,
            color=PALETTE["known"],
            alpha=0.82,
            linewidth=0,
        )
        ax_objects.barh(
            y,
            new,
            left=known,
            height=0.42,
            color=PALETTE["new"],
            alpha=0.95,
            linewidth=0,
        )
        ax_objects.text(
            known / 2,
            y,
            f"{known:,}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
        ax_objects.text(
            known + new + max_total * 0.025,
            y,
            f"+{new:,}",
            ha="left",
            va="center",
            color=PALETTE["dark"],
            fontsize=8,
            fontweight="bold",
        )
    ax_objects.set(
        title="对象覆盖",
        yticks=y_positions,
        yticklabels=[item[0] for item in object_rows],
        xlim=(0, max_total * 1.18),
        ylim=(-0.55, 1.55),
        xlabel="唯一对象数量",
    )
    ax_objects.tick_params(axis="y", length=0, pad=5)
    ax_objects.spines["left"].set_visible(False)
    ax_objects.spines["bottom"].set_color("#C9D6D2")
    ax_objects.text(
        0,
        1.42,
        "灰蓝：训练中已见    黄：正式测试新增",
        transform=ax_objects.transData,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["dark"],
    )

    add_panel_labels([ax_matrix, ax_objects])
    output_stem = FIGURES_DIR / "raw_q2_cold_start_structure"
    outputs = export_figure(fig, output_stem, dpi=600, grayscale_preview=True)
    fig.savefig(output_stem.with_suffix(".svg"), format="svg")
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    result = draw()
    for key, path in result.items():
        print(f"{key}: {path}")
