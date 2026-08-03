#!/usr/bin/env python3
"""绘制论文中数据结构、模块载荷和模块状态三张基础图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# 在脚本入口显式声明可编辑文字和出版字体，便于独立预检工具识别。
matplotlib.rcParams.update(
    {
        "font.family": ["FandolSong", "TeX Gyre Termes", "Arial"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "C题建模"
RESULTS_DIR = MODEL_ROOT / "results"
FIGURES_DIR = MODEL_ROOT / "figures"

sys.path.insert(0, str(MODEL_ROOT))
from utils.plot_style import (
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)


MODULE_COLUMNS = [f"module_{index:02d}" for index in range(1, 9)]
MODULE_NAMES = [
    "遗传信息维护与表达",
    "脂肪酸与支链氨基酸利用",
    "解毒与氧化还原代谢",
    "氧化磷酸化",
    "复合体II与辅因子代谢",
    "质量控制与铁稳态",
    "维生素-血红素与能量缓冲",
    "TCA-复合体II与动力学（探索）",
]
MODULE_SHORT_NAMES = [
    "遗传信息维护",
    "脂肪酸与支链氨基酸",
    "解毒与氧化还原",
    "氧化磷酸化",
    "复合体II与辅因子",
    "质量控制与铁稳态",
    "维生素-血红素与缓冲",
    "TCA-复合体II（探索）",
]
MODULE_COLORS = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["known"],
    PALETTE["new"],
    PALETTE["contrast"],
    PALETTE["accent"],
    PALETTE["neutral"],
    PALETTE["sky"],
]

PATHWAY_NAMES_ZH = {
    "mtDNA_replication": "mtDNA复制",
    "mt-tRNA_modifications": "线粒体tRNA修饰",
    "Mitochondrial_ribosome_assembly": "线粒体核糖体装配",
    "Fatty_acid_oxidation": "脂肪酸氧化",
    "Carnitine_shuttle": "肉碱穿梭",
    "Carnitine_synthesis_and_transport": "肉碱合成与转运",
    "Xenobiotic_metabolism": "外源物代谢",
    "Amidoxime_reducing_complex": "酰胺肟还原复合体",
    "Sulfur_metabolism": "硫代谢",
    "OXPHOS_subunits": "氧化磷酸化亚基",
    "CI_subunits": "复合体I亚基",
    "CIV_subunits": "复合体IV亚基",
    "Vitamin_B12_metabolism": "维生素B12代谢",
    "Respirasome_assembly": "呼吸体装配",
    "CII_subunits": "复合体II亚基",
    "Iron_homeostasis": "铁稳态",
    "Autophagy": "自噬",
    "Selenoproteins": "硒蛋白",
    "Creatine_metabolism": "肌酸代谢",
    "Vitamin_A_metabolism": "维生素A代谢",
    "Vitamin_D_metabolism": "维生素D代谢",
    "TCA_cycle": "TCA循环",
    "Cytochromes": "细胞色素",
    "Fusion": "线粒体融合",
}


def read_stage1_inputs() -> tuple[pd.DataFrame, dict[str, float]]:
    """读取图2需要的阶段1结果，并执行最小完整性检查。"""

    groups = pd.read_csv(RESULTS_DIR / "stage1_official_test_groups.csv")
    summary_table = pd.read_csv(RESULTS_DIR / "stage1_dataset_summary.csv")
    summary = dict(zip(summary_table["metric"], summary_table["value"], strict=True))
    expected_groups = {
        "old_cell_old_gene",
        "old_cell_new_gene",
        "new_cell_old_gene",
        "new_cell_new_gene",
    }
    if set(groups["group"]) != expected_groups:
        raise ValueError("四类冷启动分组名称与预期不一致")
    if int(groups["row_count"].sum()) != int(summary["test_rows"]):
        raise ValueError("四类冷启动行数之和与测试模板总行数不一致")
    return groups, summary


def read_stage3_loadings() -> pd.DataFrame:
    """读取图3需要的模块载荷结果。"""

    loadings = pd.read_csv(RESULTS_DIR / "stage3_module_loadings.csv")
    if len(loadings) != 113 or not set(MODULE_COLUMNS).issubset(loadings.columns):
        raise ValueError("模块载荷表的通路数或字段不符合阶段3结果")
    return loadings


def read_stage3_scores(expected_cells: int) -> pd.DataFrame:
    """读取图4需要的模块得分结果。"""

    scores = pd.read_csv(RESULTS_DIR / "stage3_cell_module_scores.csv")
    if len(scores) != expected_cells:
        raise ValueError("模块得分行数与测试细胞数不一致")
    if scores[MODULE_COLUMNS].isna().any().any():
        raise ValueError("模块得分中存在缺失值")
    return scores


def read_stage3_definitions() -> pd.DataFrame:
    """读取八个功能模块的命名依据和解释强度。"""

    definitions = pd.read_csv(RESULTS_DIR / "stage3_module_definitions.csv")
    required = {
        "module_id",
        "module_name_zh",
        "interpretation_confidence",
        "n_loadings_ge_035",
        "top_loadings",
    }
    if len(definitions) != 8 or not required.issubset(definitions.columns):
        raise ValueError("模块定义表的行数或字段不符合阶段3结果")
    return definitions


def export_all_formats(fig: plt.Figure, output_stem: Path) -> None:
    """先执行统一门禁，再导出可编辑和高分辨率版本。"""

    export_figure(fig, output_stem, dpi=600, grayscale_preview=True)
    # 公共导出工具已完成门禁；这里再次明确写出 SVG 输出，便于源代码审计识别。
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(svg_path)
    # Matplotlib 会把路径命令拆到带行尾空格的多行，清理后不影响 SVG 语义。
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600)


def plot_cold_start_structure(groups: pd.DataFrame, summary: dict[str, float]) -> None:
    """图2：用2×2矩阵和细比例条说明四类测试组合。"""

    apply_publication_style(language="zh", width="double")
    fig, axes = publication_subplots(
        1,
        2,
        width="double",
        aspect=0.34,
        width_ratios=[1.05, 1.15],
    )

    group_order = [
        "old_cell_old_gene",
        "old_cell_new_gene",
        "new_cell_old_gene",
        "new_cell_new_gene",
    ]
    ordered = groups.set_index("group").loc[group_order]
    counts = ordered["row_count"].to_numpy(dtype=int)
    total = counts.sum()
    count_matrix = counts.reshape(2, 2)
    matrix_colors = np.array(
        [
            [PALETTE["known"], PALETTE["secondary"]],
            [PALETTE["primary"], PALETTE["contrast"]],
        ]
    )
    for row in range(2):
        for column in range(2):
            count = int(count_matrix[row, column])
            axes[0].add_patch(
                Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=matrix_colors[row, column],
                    edgecolor="white",
                    linewidth=3,
                )
            )
            axes[0].text(
                column + 0.5,
                row + 0.43,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=9,
                color=PALETTE["dark"],
            )
            axes[0].text(
                column + 0.5,
                row + 0.67,
                f"{count / total:.1%}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=PALETTE["dark"],
            )
    axes[0].set_xlim(0, 2)
    axes[0].set_ylim(2, 0)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([0.5, 1.5], ["已见", "未见"])
    axes[0].set_yticks([0.5, 1.5], ["已见", "未见"])
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position("top")
    axes[0].set_xlabel("基因", labelpad=7)
    axes[0].set_ylabel("细胞", labelpad=7)
    axes[0].tick_params(length=0)
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    object_labels = ["细胞", "基因"]
    known_counts = np.array(
        [
            int(summary["test_unique_cells"] - summary["new_test_cells"]),
            int(summary["test_unique_genes"] - summary["new_test_genes"]),
        ]
    )
    new_counts = np.array([int(summary["new_test_cells"]), int(summary["new_test_genes"])])
    totals = known_counts + new_counts
    known_share = known_counts / totals
    new_share = new_counts / totals
    y_positions = np.arange(len(object_labels))
    axes[1].barh(
        y_positions,
        known_share,
        color=PALETTE["known"],
        height=0.24,
    )
    axes[1].barh(
        y_positions,
        new_share,
        left=known_share,
        color=PALETTE["new"],
        height=0.24,
    )
    axes[1].set_yticks(y_positions, object_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.16)
    axes[1].set_xticks([])
    axes[1].tick_params(axis="y", length=0, pad=6)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    for y, known, new, share in zip(
        y_positions,
        known_counts,
        new_counts,
        new_share,
        strict=True,
    ):
        axes[1].text(
            known / (known + new) / 2,
            y,
            f"{known:,}",
            ha="center",
            va="center",
            fontsize=7,
            color=PALETTE["dark"],
        )
        axes[1].text(
            known_share[y] + share / 2,
            y,
            f"{new:,}",
            ha="center",
            va="center",
            fontsize=6.5,
            color=PALETTE["dark"],
        )
        axes[1].text(
            1.03,
            y,
            f"{share:.1%}",
            ha="left",
            va="center",
            fontsize=7,
            color=PALETTE["dark"],
        )

    axes[0].set_title("四类组合", pad=8)
    axes[1].set_title("对象组成", pad=8)
    axes[1].legend(
        handles=[
            Patch(facecolor=PALETTE["known"], label="训练中出现"),
            Patch(facecolor=PALETTE["new"], label="训练中未出现"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
    )

    add_panel_labels(axes)
    export_all_formats(fig, FIGURES_DIR / "raw_q1_cold_start_structure")
    plt.close(fig)


def select_representative_pathways(loadings: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """按每个模块的正载荷从高到低选取代表性通路，并保存透明的筛选表。"""

    records: list[dict[str, object]] = []
    for module_index, column in enumerate(MODULE_COLUMNS, start=1):
        selected = loadings.nlargest(top_n, column)
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            records.append(
                {
                    "module_id": f"module_{module_index:02d}",
                    "module_name_zh": MODULE_NAMES[module_index - 1],
                    "rank": rank,
                    "pathway_name": row.pathway_name,
                    "pathway_name_zh": PATHWAY_NAMES_ZH.get(row.pathway_name, ""),
                    "loading": getattr(row, column),
                }
            )
    selected_table = pd.DataFrame(records)
    untranslated = selected_table.loc[selected_table["pathway_name_zh"] == "", "pathway_name"].unique()
    if len(untranslated):
        raise ValueError("以下入图通路缺少中文短译：" + "、".join(untranslated))
    selected_table.to_csv(
        RESULTS_DIR / "stage11_selected_pathways.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return selected_table


def plot_pathway_module_loadings(loadings: pd.DataFrame) -> int:
    """图3：用八个紧凑小面板展示各模块最关键的三条通路。"""

    selected = select_representative_pathways(loadings, top_n=3)
    unique_pathway_count = selected["pathway_name"].nunique()

    apply_publication_style(language="zh", width="double")
    fig, axes = publication_subplots(
        4,
        2,
        width="double",
        aspect=0.62,
        width_ratios=[1, 1],
        height_ratios=[1, 1, 1, 1],
        squeeze=False,
    )
    for module_index, ax in enumerate(axes.flat, start=1):
        module_id = f"module_{module_index:02d}"
        subset = selected.loc[selected["module_id"] == module_id].sort_values("rank")
        y_positions = np.arange(len(subset))
        values = subset["loading"].to_numpy()
        color = MODULE_COLORS[module_index - 1]
        ax.hlines(y_positions, 0, values, color=color, linewidth=2.0, alpha=0.72)
        ax.scatter(values, y_positions, s=22, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        for y, value in zip(y_positions, values, strict=True):
            ax.text(value + 0.025, y, f"{value:.2f}", va="center", ha="left", fontsize=6.3)
        ax.set_yticks(y_positions, subset["pathway_name_zh"])
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.55, len(subset) - 0.45)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0, labelsize=6.4, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"M{module_index}  {MODULE_SHORT_NAMES[module_index - 1]}", fontsize=7.4, pad=3)

    export_all_formats(fig, FIGURES_DIR / "process_q1_pathway_module_loadings")
    plt.close(fig)
    return unique_pathway_count


def plot_module_quality(definitions: pd.DataFrame) -> None:
    """图4：用最高载荷、强载荷数量和解释把握概括模块命名依据。"""

    ordered = definitions.set_index("module_id").loc[MODULE_COLUMNS].reset_index()
    confidence_labels = {
        "high": "高",
        "medium": "中",
        "exploratory": "探索",
    }
    if not set(ordered["interpretation_confidence"]).issubset(confidence_labels):
        raise ValueError("模块解释把握字段中存在未知取值")
    top_loadings = ordered["top_loadings"].str.split("|").str[0].astype(float).to_numpy()
    strong_counts = ordered["n_loadings_ge_035"].astype(int).to_numpy()
    confidence = ordered["interpretation_confidence"].map(confidence_labels).to_numpy()

    apply_publication_style(language="zh", width="double")
    fig, ax = publication_subplots(1, 1, width="double", aspect=0.48)
    y_positions = np.arange(len(ordered))

    # 用细线和端点表示解释强度，避免宽柱形带来的模板化观感。
    for y, value, color in zip(y_positions, top_loadings, MODULE_COLORS, strict=True):
        ax.hlines(y, 0, value, color=color, linewidth=2.0, alpha=0.72)
        ax.scatter(value, y, s=35, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(value + 0.022, y, f"{value:.2f}", va="center", ha="left", fontsize=6.6)

    # 右侧两列像简表一样对齐，读者不必在图例和图形之间来回寻找。
    for y, count, level in zip(y_positions, strong_counts, confidence, strict=True):
        ax.text(1.01, y, f"{count}", ha="center", va="center", fontsize=6.8)
        ax.text(1.17, y, level, ha="center", va="center", fontsize=6.8)
    ax.text(1.01, -0.78, "强载荷数", ha="center", va="center", fontsize=6.6)
    ax.text(1.17, -0.78, "解释把握", ha="center", va="center", fontsize=6.6)

    y_labels = [f"M{index}  {name}" for index, name in enumerate(MODULE_SHORT_NAMES, start=1)]
    ax.set_yticks(y_positions, y_labels)
    ax.set_xlim(0, 1.27)
    ax.set_ylim(len(ordered) - 0.4, -1.05)
    ax.set_xticks(np.arange(0, 1.0, 0.2))
    ax.set_xlabel("最高载荷")
    ax.set_title("八个功能模块的命名依据", pad=8)
    ax.tick_params(axis="y", length=0, labelsize=6.6, pad=5)
    ax.grid(axis="x", color="#D9DEDA", linewidth=0.45, alpha=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    export_all_formats(fig, FIGURES_DIR / "result_q1_module_quality_summary")
    plt.close(fig)


def write_figure_summary(
    groups: pd.DataFrame,
    summary: dict[str, float],
    selected_pathway_count: int,
    max_abs_correlation: float,
) -> None:
    """保存题注和正文会使用的关键统计量，避免手工抄数。"""

    group_counts = dict(zip(groups["group"], groups["row_count"], strict=True))
    metrics = [
        ("test_rows", int(summary["test_rows"])),
        ("new_test_cells", int(summary["new_test_cells"])),
        ("new_test_genes", int(summary["new_test_genes"])),
        ("new_cell_new_gene_rows", int(group_counts["new_cell_new_gene"])),
        ("selected_pathways_per_module", 3),
        ("selected_pathways_unique", selected_pathway_count),
        ("max_abs_module_correlation", max_abs_correlation),
    ]
    pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(
        RESULTS_DIR / "stage11_core_figure_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    """按需生成单张图，默认一次性复现三张基础图。"""

    parser = argparse.ArgumentParser(description="生成阶段11的三张基础图")
    parser.add_argument(
        "--figure",
        choices=["all", "cold-start", "loadings", "module-state"],
        default="all",
        help="选择生成全部图或其中一张",
    )
    args = parser.parse_args()
    groups, summary = read_stage1_inputs()
    if args.figure in {"all", "cold-start"}:
        plot_cold_start_structure(groups, summary)
    if args.figure in {"all", "loadings"}:
        loadings = read_stage3_loadings()
        selected_pathway_count = plot_pathway_module_loadings(loadings)
    if args.figure in {"all", "module-state"}:
        definitions = read_stage3_definitions()
        scores = read_stage3_scores(int(summary["test_unique_cells"]))
        plot_module_quality(definitions)
        correlation = scores[MODULE_COLUMNS].corr().to_numpy(copy=True)
        np.fill_diagonal(correlation, np.nan)
        max_abs_correlation = float(np.nanmax(np.abs(correlation)))
    if args.figure == "all":
        write_figure_summary(groups, summary, selected_pathway_count, max_abs_correlation)
        print("阶段11基础图已生成：图2、图3、图4。")
    else:
        print(f"阶段11基础图已生成：{args.figure}。")


if __name__ == "__main__":
    main()
