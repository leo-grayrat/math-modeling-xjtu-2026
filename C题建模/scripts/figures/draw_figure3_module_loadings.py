#!/usr/bin/env python3
"""绘制图 3：通路载荷热图与模块质量摘要。"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import (  # noqa: E402
    CONFIDENCE_COLORS,
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)


MODULES = [f"module_{number:02d}" for number in range(1, 9)]
SHORT_NAMES = [f"M{number}" for number in range(1, 9)]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取载荷矩阵和模块摘要，并核对模块覆盖。"""

    loadings = pd.read_csv(RESULTS_DIR / "stage3_module_loadings.csv")
    definitions = pd.read_csv(RESULTS_DIR / "stage3_module_definitions.csv")
    if list(loadings.columns[2:]) != MODULES:
        raise ValueError("载荷矩阵的模块列顺序与结果契约不一致")
    if list(definitions["module_id"]) != MODULES:
        raise ValueError("模块摘要缺少或打乱了八个模块")
    if len(loadings) != 113:
        raise ValueError("末级通路数量不是 113，拒绝静默换用其他版本")
    return loadings, definitions


def select_pathways(loadings: pd.DataFrame) -> pd.DataFrame:
    """按每个模块的绝对载荷前四名生成可读的去重展示集合。"""

    selected: set[str] = set()
    for module in MODULES:
        ranked = loadings.loc[loadings[module].abs().nlargest(4).index, "pathway_name"]
        selected.update(ranked.tolist())
    chosen = loadings[loadings["pathway_name"].isin(selected)].copy()
    chosen["max_abs_loading"] = chosen[MODULES].abs().max(axis=1)
    chosen = chosen.sort_values("max_abs_loading", ascending=True)
    chosen["display_name"] = chosen["pathway_name"].str.replace("_", " ", regex=False)
    chosen["display_name"] = chosen["display_name"].map(
        lambda text: "\n".join(textwrap.wrap(text, width=25))
    )
    return chosen


def draw() -> dict[str, str]:
    apply_publication_style(language="zh", width="report")
    # 把关键出版设置显式放在入口，便于静态预检和复现。
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["FandolSong", "Times New Roman", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    loadings, definitions = load_data()
    chosen = select_pathways(loadings)

    diverging = LinearSegmentedColormap.from_list(
        "blue_white_coral",
        [PALETTE["contrast"], "#FFFFFF", PALETTE["risk"]],
    )
    heat_values = chosen[MODULES].to_numpy(dtype=float)
    max_abs = float(np.abs(heat_values).max())
    norm = Normalize(vmin=-max_abs, vmax=max_abs)

    fig, axes = publication_subplots(
        1,
        2,
        width="report",
        aspect=0.86,
        width_ratios=[1.7, 0.85],
    )
    ax_heat, ax_quality = axes

    image = ax_heat.imshow(
        heat_values,
        cmap=diverging,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax_heat.set(
        title="通路载荷",
        xticks=np.arange(len(MODULES)),
        xticklabels=SHORT_NAMES,
        yticks=np.arange(len(chosen)),
        yticklabels=chosen["display_name"],
    )
    ax_heat.tick_params(axis="x", length=0, pad=4, labelsize=6.6)
    ax_heat.tick_params(axis="y", length=0, pad=4, labelsize=6.0)
    ax_heat.spines[:].set_visible(False)
    for row_index in range(len(chosen) + 1):
        ax_heat.axhline(row_index - 0.5, color="white", linewidth=0.8, zorder=2)
    for col_index in range(len(MODULES) + 1):
        ax_heat.axvline(col_index - 0.5, color="white", linewidth=0.8, zorder=2)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=diverging),
        ax=ax_heat,
        fraction=0.045,
        pad=0.02,
    )
    colorbar.set_label("载荷", rotation=90, labelpad=5)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2, width=0.5, labelsize=6.2)

    quality = definitions.copy()
    y = np.arange(len(quality))[::-1]
    purity = quality["top8_category_purity"].to_numpy(dtype=float)
    count = quality["n_loadings_ge_035"].to_numpy(dtype=float)
    sizes = 120 + 520 * np.sqrt(count / count.max())
    confidence_colors = CONFIDENCE_COLORS
    confidence_values = quality["interpretation_confidence"].tolist()
    confidence_colors_for_rows = [confidence_colors[value] for value in confidence_values]
    ax_quality.scatter(
        purity,
        y,
        s=sizes,
        c=confidence_colors_for_rows,
        alpha=0.94,
        edgecolors="white",
        linewidths=1.0,
        zorder=2,
    )
    # 数字放在对应圆点内，只标这一项的强载荷通路数，避免另起一张低信息量表格。
    for row_y, row_purity, row_count, row_color in zip(y, purity, count, confidence_colors_for_rows):
        text_color = "white" if row_color in {PALETTE["primary"], PALETTE["secondary"]} else PALETTE["dark"]
        ax_quality.text(
            row_purity,
            row_y,
            f"{int(row_count)}",
            ha="center",
            va="center",
            fontsize=6.6,
            color=text_color,
            zorder=3,
        )
    short_module_names = [
        "线粒体表达",
        "脂肪酸/BCAA",
        "解毒与氧化还原",
        "氧化磷酸化",
        "复合体 II",
        "质量控制",
        "维生素/血红素",
        "TCA-动力学",
    ]
    ax_quality.set(
        title="模块载荷广度",
        xlabel="前八条通路类别集中度\n圆内数字表示强载荷通路数",
        yticks=y,
        yticklabels=[f"M{number}  {name}" for number, name in zip(range(1, 9), short_module_names)],
        xlim=(0.25, 1.10),
        ylim=(-0.7, len(quality) - 0.3),
    )
    ax_quality.tick_params(axis="y", length=0, pad=4)
    ax_quality.spines["left"].set_visible(False)
    ax_quality.spines["bottom"].set_color("#C9D6D2")
    ax_quality.set_xticks([0.4, 0.6, 0.8, 1.0])
    for guide in [0.4, 0.6, 0.8, 1.0]:
        ax_quality.axvline(guide, color="#E8EFEC", linewidth=0.7, zorder=0)
    ax_quality.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=color,
                   markeredgecolor="white", label=label)
            for label, color in [
                ("高", confidence_colors["high"]),
                ("中", confidence_colors["medium"]),
                ("低/探索", confidence_colors["exploratory"]),
            ]
        ],
        title="解释把握",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=6.3,
        title_fontsize=6.3,
        handletextpad=0.35,
        borderpad=0.1,
    )
    add_panel_labels([ax_heat, ax_quality])
    output_stem = FIGURES_DIR / "result_q1_module_loadings_quality"
    outputs = export_figure(fig, output_stem, dpi=600, grayscale_preview=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    result = draw()
    for key, path in result.items():
        print(f"{key}: {path}")
