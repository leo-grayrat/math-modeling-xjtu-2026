#!/usr/bin/env python3
"""绘制图11：三个虚拟敲除案例的通路、模块与预测响应。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib as mpl

# 终端运行时不打开交互窗口。
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties, fontManager
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


CASE_ORDER = ["HSPE1", "PRELID1", "ISCU"]
PATHWAY_ORDER = [
    "Chaperones",
    "Protein_homeostasis",
    "Protein_import_sorting_and_homeostasis",
    "Cardiolipin_synthesis",
    "Phospholipid_metabolism",
    "Lipid_metabolism",
    "Metabolism",
    "Fe-S_cluster_biosynthesis",
    "Fe-S-containing_proteins",
    "Metals_and_cofactors",
]
PATHWAY_LABELS = {
    "Chaperones": "伴侣蛋白",
    "Protein_homeostasis": "蛋白质稳态",
    "Protein_import_sorting_and_homeostasis": "蛋白质导入、分选与稳态",
    "Cardiolipin_synthesis": "心磷脂合成",
    "Phospholipid_metabolism": "磷脂代谢",
    "Lipid_metabolism": "脂质代谢",
    "Metabolism": "代谢",
    "Fe-S_cluster_biosynthesis": "铁硫簇生物合成",
    "Fe-S-containing_proteins": "含铁硫蛋白",
    "Metals_and_cofactors": "金属与辅因子",
}
MODULE_ORDER = [f"module_{index:02d}" for index in range(1, 9)]
MODULE_LABELS = [
    "M1 综合状态",
    "M2 脂肪酸利用",
    "M3 解毒与氧化还原",
    "M4 氧化磷酸化",
    "M5 复合体II与辅因子",
    "M6 质量控制与铁稳态",
    "M7 维生素与能量缓冲",
    "M8 TCA与动力学",
]
LINE_COLORS = {
    "HSPE1": PALETTE["risk"],
    "PRELID1": PALETTE["positive"],
    "ISCU": PALETTE["known"],
}
LINE_MARKERS = {"HSPE1": "v", "PRELID1": "^", "ISCU": "s"}
LINE_STYLES = {"HSPE1": "-", "PRELID1": "-", "ISCU": "--"}


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取并核对三个案例的阶段9正式结果。"""

    pathways = pd.read_csv(RESULTS_DIR / "stage9_case_pathway_changes.csv")
    modules = pd.read_csv(RESULTS_DIR / "stage9_case_module_changes.csv")
    predictions = pd.read_csv(RESULTS_DIR / "stage9_counterfactual_predictions.csv")
    summary = pd.read_csv(RESULTS_DIR / "stage9_case_summary.csv")

    required = {
        "pathways": {
            "case_id",
            "cell_line_id",
            "gene",
            "intensity",
            "pathway_name",
            "pathway_change",
        },
        "modules": {
            "case_id",
            "cell_line_id",
            "gene",
            "intensity",
            "module_name",
            "module_change",
        },
        "predictions": {
            "case_id",
            "cell_line_id",
            "gene",
            "intensity",
            "prediction_change",
        },
        "summary": {
            "case_id",
            "cell_line_id",
            "gene",
            "baseline_prediction",
            "pathway_count",
        },
    }
    frames = {
        "pathways": pathways,
        "modules": modules,
        "predictions": predictions,
        "summary": summary,
    }
    for name, frame in frames.items():
        missing = required[name] - set(frame.columns)
        if missing:
            raise ValueError(f"阶段9的{name}数据缺少字段：{sorted(missing)}")
        if set(frame["gene"].unique()) != set(CASE_ORDER):
            raise ValueError(f"阶段9的{name}数据案例与图表契约不一致")

    if len(summary) != 3 or summary["gene"].duplicated().any():
        raise ValueError("阶段9案例汇总必须恰好包含三个不重复案例")
    expected_intensities = {0.5, 0.75, 1.0}
    for name, frame in (("通路", pathways), ("模块", modules), ("预测", predictions)):
        if set(frame["intensity"].unique()) != expected_intensities:
            raise ValueError(f"{name}数据必须包含50%、75%和100%三档扰动")
    if not np.isfinite(pathways["pathway_change"]).all():
        raise ValueError("通路变化存在缺失或无穷值")
    if not np.isfinite(modules["module_change"]).all():
        raise ValueError("模块变化存在缺失或无穷值")
    if not np.isfinite(predictions["prediction_change"]).all():
        raise ValueError("预测变化存在缺失或无穷值")

    full_pathways = pathways[pathways["intensity"].eq(1.0)].copy()
    affected = full_pathways.groupby("pathway_name")["pathway_change"].apply(
        lambda values: bool(np.any(~np.isclose(values.to_numpy(dtype=float), 0.0)))
    )
    affected_names = set(affected[affected].index)
    if affected_names != set(PATHWAY_ORDER):
        raise ValueError("完全扰动下的10条受影响通路与图表契约不一致")

    full_modules = modules[modules["intensity"].eq(1.0)]
    expected_module_pairs = {(gene, module) for gene in CASE_ORDER for module in MODULE_ORDER}
    actual_module_pairs = set(
        full_modules[["gene", "module_name"]].itertuples(index=False, name=None)
    )
    if actual_module_pairs != expected_module_pairs or len(full_modules) != 24:
        raise ValueError("完全扰动下必须包含三个案例各八个模块")
    if len(predictions) != 9:
        raise ValueError("三个案例必须各有三档扰动预测")
    return pathways, modules, predictions, summary


def build_heatmap_matrices(
    pathways: pd.DataFrame,
    modules: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """整理完全扰动时的通路与模块变化矩阵。"""

    pathway_matrix = (
        pathways[pathways["intensity"].eq(1.0)]
        .pivot(index="pathway_name", columns="gene", values="pathway_change")
        .reindex(index=PATHWAY_ORDER, columns=CASE_ORDER)
        .to_numpy(dtype=float)
    )
    pathway_matrix[np.isclose(pathway_matrix, 0.0)] = np.nan
    module_matrix = (
        modules[modules["intensity"].eq(1.0)]
        .pivot(index="module_name", columns="gene", values="module_change")
        .reindex(index=MODULE_ORDER, columns=CASE_ORDER)
        .to_numpy(dtype=float)
    )
    return pathway_matrix, module_matrix


def add_panel_label(
    axis: plt.Axes,
    label: str,
    font: FontProperties,
    x: float = -0.04,
) -> None:
    """在面板左上角添加小写编号。"""

    axis.text(
        x,
        1.07,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=font,
        fontsize=8.5,
        color=PALETTE["dark"],
        clip_on=False,
    )


def draw_heatmap(
    axis: plt.Axes,
    colorbar_axis: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    title: str,
    limit: float,
    zh_font: FontProperties,
    en_font: FontProperties,
    panel_font: FontProperties,
    panel_label: str,
    xlabels_top: bool,
) -> None:
    """绘制带固定零点和缺失掩膜的方向热图。"""

    color_map = LinearSegmentedColormap.from_list(
        "project_diverging",
        [PALETTE["contrast"], "#FBFCFA", PALETTE["risk"]],
    )
    color_map.set_bad("#F0F2F1")
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap=color_map,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(CASE_ORDER)))
    axis.set_xticklabels(CASE_ORDER, fontproperties=en_font, fontsize=7.3)
    axis.tick_params(
        axis="x",
        top=xlabels_top,
        labeltop=xlabels_top,
        bottom=not xlabels_top,
        labelbottom=not xlabels_top,
        length=0,
        pad=4,
    )
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels, fontproperties=zh_font, fontsize=6.7)
    axis.tick_params(axis="y", length=0, pad=5)
    axis.set_xticks(np.arange(-0.5, len(CASE_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, loc="left", fontproperties=zh_font, fontsize=7.8, pad=6)
    add_panel_label(axis, panel_label, panel_font)

    colorbar = axis.figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks([-limit, 0.0, limit])
    decimals = 2 if limit < 0.5 else 1
    colorbar.set_ticklabels(
        [f"−{limit:.{decimals}f}", "0", f"{limit:.{decimals}f}"]
    )
    colorbar.ax.tick_params(length=2.5, width=0.5, pad=2)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontproperties(en_font)
        tick.set_fontsize(6.2)
    colorbar.outline.set_visible(False)


def draw_prediction_panel(
    axis: plt.Axes,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    zh_font: FontProperties,
    en_font: FontProperties,
    panel_font: FontProperties,
) -> None:
    """绘制四档扰动下的预测变化，并在曲线末端直接标记基因。"""

    baseline = summary[["case_id", "cell_line_id", "gene"]].copy()
    baseline["intensity"] = 0.0
    baseline["prediction_change"] = 0.0
    curve_data = pd.concat(
        [
            baseline[["case_id", "cell_line_id", "gene", "intensity", "prediction_change"]],
            predictions[["case_id", "cell_line_id", "gene", "intensity", "prediction_change"]],
        ],
        ignore_index=True,
    )

    for gene in CASE_ORDER:
        series = curve_data[curve_data["gene"].eq(gene)].sort_values("intensity")
        x_values = series["intensity"].to_numpy(dtype=float) * 100
        y_values = series["prediction_change"].to_numpy(dtype=float)
        color = LINE_COLORS[gene]
        axis.plot(
            x_values,
            y_values,
            color=color,
            linestyle=LINE_STYLES[gene],
            linewidth=1.45,
            marker=LINE_MARKERS[gene],
            markersize=4.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
        y_offset = {"HSPE1": -0.0012, "PRELID1": 0.0008, "ISCU": 0.0015}[gene]
        axis.text(
            103.5,
            y_values[-1] + y_offset,
            gene,
            ha="left",
            va="center",
            fontproperties=en_font,
            fontsize=7.2,
            color=color,
            clip_on=False,
        )

    axis.axhline(0, color="#8D969C", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    axis.set_xlim(0, 118)
    axis.set_ylim(-0.022, 0.0225)
    axis.set_xticks([0, 50, 75, 100])
    axis.set_xticklabels(["0", "50", "75", "100"], fontproperties=en_font, fontsize=7.0)
    axis.set_yticks([-0.02, -0.01, 0.0, 0.01, 0.02])
    axis.set_yticklabels(
        ["−0.02", "−0.01", "0", "0.01", "0.02"],
        fontproperties=en_font,
        fontsize=7.0,
    )
    axis.set_xlabel("输入移除比例（%）", fontproperties=zh_font, fontsize=7.5, labelpad=5)
    axis.set_ylabel("预测变化", fontproperties=zh_font, fontsize=7.5, labelpad=6)
    axis.tick_params(axis="both", length=3, width=0.6, pad=3)
    axis.grid(axis="y", color="#E6EAE7", linewidth=0.55, zorder=0)
    axis.spines["left"].set_color(PALETTE["dark"])
    axis.spines["bottom"].set_color(PALETTE["dark"])
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.set_title("预测响应", loc="left", fontproperties=zh_font, fontsize=7.8, pad=6)
    add_panel_label(axis, "c", panel_font, x=-0.06)


def draw_figure() -> tuple[plt.Figure, dict[str, float]]:
    """按照三面板不对称契约绘制图11。"""

    apply_publication_style(language="zh", width="double")
    mpl.rcParams["figure.constrained_layout.use"] = False
    zh_font, en_font, en_bold_font = configure_fonts()
    pathways, modules, predictions, summary = load_data()
    pathway_matrix, module_matrix = build_heatmap_matrices(pathways, modules)

    figure = plt.figure(figsize=(7.2, 4.35), layout=None)
    main_grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.62, 1.0],
        height_ratios=[1.25, 1.0],
        hspace=0.42,
        wspace=0.37,
    )
    upper_grid = main_grid[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.06)
    lower_grid = main_grid[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.06)
    pathway_axis = figure.add_subplot(upper_grid[0, 0])
    pathway_colorbar = figure.add_subplot(upper_grid[0, 1])
    module_axis = figure.add_subplot(lower_grid[0, 0])
    module_colorbar = figure.add_subplot(lower_grid[0, 1])
    prediction_axis = figure.add_subplot(main_grid[:, 1])
    figure.subplots_adjust(left=0.235, right=0.965, bottom=0.14, top=0.90)

    draw_heatmap(
        pathway_axis,
        pathway_colorbar,
        pathway_matrix,
        [PATHWAY_LABELS[name] for name in PATHWAY_ORDER],
        "完全扰动下的通路变化",
        0.9,
        zh_font,
        en_font,
        en_bold_font,
        "a",
        True,
    )
    draw_heatmap(
        module_axis,
        module_colorbar,
        module_matrix,
        MODULE_LABELS,
        "完全扰动下的模块变化",
        0.45,
        zh_font,
        en_font,
        en_bold_font,
        "b",
        False,
    )
    draw_prediction_panel(
        prediction_axis,
        predictions,
        summary,
        zh_font,
        en_font,
        en_bold_font,
    )

    layout_issues = audit_layout(figure)
    design_issues = audit_design(figure)
    if layout_issues or design_issues:
        raise ValueError("图11预检未通过：" + "；".join(layout_issues + design_issues))

    summary_values = {
        "affected_pathways": float(np.isfinite(pathway_matrix).sum()),
        "max_abs_pathway_change": float(np.nanmax(np.abs(pathway_matrix))),
        "max_abs_module_change": float(np.max(np.abs(module_matrix))),
        "max_abs_prediction_change": float(np.max(np.abs(predictions["prediction_change"]))),
    }
    return figure, summary_values


def export_preview(figure: plt.Figure) -> Path:
    """导出一张供交互审阅的PNG。"""

    output = FIGURES_DIR / "result_q2_counterfactual_chain_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    return output


def export_final(figure: plt.Figure) -> list[Path]:
    """导出论文使用的矢量图、位图和灰度检查图。"""

    stem = FIGURES_DIR / "result_q2_counterfactual_chain"
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
    """执行选定导出模式并打印数据量级。"""

    args = parse_args()
    figure, summary_values = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    print(
        "图11数据核对："
        f"非空通路单元{int(summary_values['affected_pathways'])}个，"
        f"最大通路变化{summary_values['max_abs_pathway_change']:.6f}，"
        f"最大模块变化{summary_values['max_abs_module_change']:.6f}，"
        f"最大预测变化{summary_values['max_abs_prediction_change']:.6f}。"
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
