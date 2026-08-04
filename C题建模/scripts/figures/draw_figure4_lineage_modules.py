#!/usr/bin/env python3
"""绘制图4：不同肿瘤谱系的八维功能模块平均状态。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
METADATA_PATH = (
    REPO_ROOT
    / "2026年校赛题目"
    / "C"
    / "数据文件"
    / "metadata"
    / "cell_line_metadata.csv"
)
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import PALETTE, apply_publication_style  # noqa: E402


MODULES = [f"module_{number:02d}" for number in range(1, 9)]
MODULE_LABELS = [
    "综合状态/\n遗传表达",
    "脂肪酸/\n支链氨基酸",
    "解毒/\n氧化还原",
    "氧化\n磷酸化",
    "复合体Ⅱ/\n辅因子",
    "质量控制/\n铁稳态",
    "维生素-血红素/\n能量缓冲",
    "三羧酸-复合体Ⅱ/\n动力学",
]
LINEAGE_NAMES_ZH = {
    "Lung": "肺",
    "Lymphoid": "淋巴系统",
    "CNS/Brain": "中枢神经系统/脑",
    "Skin": "皮肤",
    "Head and Neck": "头颈",
    "Esophagus/Stomach": "食管/胃",
    "Bowel": "肠道",
    "Ovary/Fallopian Tube": "卵巢/输卵管",
    "Breast": "乳腺",
    "Bone": "骨",
    "Soft Tissue": "软组织",
    "Pancreas": "胰腺",
    "Myeloid": "髓系",
    "Peripheral Nervous System": "周围神经系统",
    "Bladder/Urinary Tract": "膀胱/泌尿道",
    "Biliary Tract": "胆道",
    "Uterus": "子宫",
    "Kidney": "肾",
    "Liver": "肝",
    "Pleura": "胸膜",
    "Cervix": "宫颈",
    "Eye": "眼",
    "Thyroid": "甲状腺",
    "Prostate": "前列腺",
}


def parse_args() -> argparse.Namespace:
    """解析预览和正式导出模式。"""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="只生成一张PNG预览")
    mode.add_argument("--final", action="store_true", help="生成论文使用的完整格式")
    return parser.parse_args()


def find_font(relative_path: str) -> Path:
    """从常见用户级字体目录定位字体，避免写死具体用户名。"""

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


def load_lineage_means() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """合并细胞得分与谱系元数据，并计算可靠谱系的平均状态。"""

    scores = pd.read_csv(RESULTS_DIR / "stage3_cell_module_scores.csv")
    metadata = pd.read_csv(METADATA_PATH)
    if len(scores) != 1140 or len(metadata) != 1140:
        raise ValueError("模块得分或细胞元数据不再是1140行")
    if list(scores.columns) != ["cell_line_id", *MODULES]:
        raise ValueError("模块得分列或M1-M8顺序与图表契约不一致")
    if scores["cell_line_id"].duplicated().any() or metadata["cell_line_id"].duplicated().any():
        raise ValueError("细胞编号存在重复，无法执行一对一合并")

    merged = scores.merge(
        metadata[["cell_line_id", "OncotreeLineage"]],
        on="cell_line_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 1140 or merged["OncotreeLineage"].isna().any():
        raise ValueError("细胞得分与谱系元数据未能完整匹配")

    all_counts = merged["OncotreeLineage"].value_counts()
    kept_lineages = all_counts[all_counts >= 10].index
    kept = merged[merged["OncotreeLineage"].isin(kept_lineages)].copy()
    kept_counts = kept.groupby("OncotreeLineage", sort=True).size()
    means = kept.groupby("OncotreeLineage", sort=True)[MODULES].mean()

    excluded_counts = all_counts[all_counts < 10]
    audit = {
        "all_cells": len(merged),
        "kept_cells": len(kept),
        "excluded_cells": int(excluded_counts.sum()),
        "kept_lineages": len(means),
        "excluded_lineages": len(excluded_counts),
    }
    expected = {
        "all_cells": 1140,
        "kept_cells": 1127,
        "excluded_cells": 13,
        "kept_lineages": 24,
        "excluded_lineages": 5,
    }
    if audit != expected:
        raise ValueError(f"谱系筛选结果发生变化：{audit}")
    missing_names = set(means.index) - set(LINEAGE_NAMES_ZH)
    if missing_names:
        raise ValueError(f"以下谱系缺少固定中文名称：{sorted(missing_names)}")
    return means, kept_counts, audit


def order_similar_lineages(means: pd.DataFrame) -> pd.DataFrame:
    """仅调整谱系行顺序，使八维平均状态相近的行彼此相邻。"""

    tree = linkage(means.to_numpy(dtype=float), method="ward", optimal_ordering=True)
    return means.iloc[leaves_list(tree)]


def text_color_for_cell(cmap, norm, value: float) -> str:
    """根据底色亮度选择极端值标注颜色。"""

    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.56 else PALETTE["dark"]


def draw_figure() -> tuple[plt.Figure, dict[str, int]]:
    """按照单面板图表契约绘制谱系模块状态热图。"""

    apply_publication_style(language="zh", width="double")
    zh_font, en_font, en_bold_font = configure_fonts()
    means, counts, audit = load_lineage_means()
    means = order_similar_lineages(means)
    counts = counts.reindex(means.index)

    color_map = LinearSegmentedColormap.from_list(
        "project_blue_white_coral",
        [PALETTE["contrast"], "#FFFFFF", PALETTE["risk"]],
    )
    norm = Normalize(vmin=-2.0, vmax=2.0, clip=True)

    fig, axis = plt.subplots(figsize=(7.2, 5.15), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.045, h_pad=0.055, wspace=0.02, hspace=0.02)
    values = means.to_numpy(dtype=float)
    image = axis.imshow(
        values,
        cmap=color_map,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_xticks(np.arange(len(MODULES)))
    axis.set_xticklabels([])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=2)
    axis.set_yticks(np.arange(len(means)))
    axis.set_yticklabels(
        [
            f"{LINEAGE_NAMES_ZH[lineage]}  ($n={int(counts[lineage])}$)"
            for lineage in means.index
        ],
        fontproperties=zh_font,
        fontsize=7.2,
    )
    axis.tick_params(axis="y", length=0, pad=5)

    for column, short_name in enumerate(MODULE_LABELS):
        axis.text(
            column,
            1.095,
            f"M{column + 1}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontproperties=en_bold_font,
            fontsize=7.6,
            color=PALETTE["dark"],
            clip_on=False,
        )
        axis.text(
            column,
            1.012,
            short_name,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontproperties=zh_font,
            fontsize=6.6,
            linespacing=1.05,
            color=PALETTE["dark"],
            clip_on=False,
        )

    for boundary in range(len(means) + 1):
        axis.axhline(boundary - 0.5, color="white", linewidth=0.65, zorder=2)
    for boundary in range(len(MODULES) + 1):
        axis.axvline(boundary - 0.5, color="white", linewidth=0.65, zorder=2)
    axis.spines[:].set_visible(False)

    for row, column in zip(*np.where(np.abs(values) >= 1.0)):
        value = values[row, column]
        axis.text(
            column,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontproperties=en_font,
            fontsize=6.5,
            color=text_color_for_cell(color_map, norm, value),
            zorder=3,
        )

    colorbar = fig.colorbar(image, ax=axis, fraction=0.030, pad=0.022, aspect=26)
    colorbar.set_label("平均模块得分", fontproperties=zh_font, fontsize=7.2, labelpad=5)
    colorbar.set_ticks([-2, -1, 0, 1, 2])
    for label in colorbar.ax.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(6.8)
    colorbar.ax.tick_params(length=2.5, width=0.55)
    colorbar.outline.set_visible(False)
    return fig, audit


def export_preview(fig: plt.Figure) -> Path:
    """只导出一张供交互审阅的PNG。"""

    output = FIGURES_DIR / "result_q1_lineage_module_state_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    return output


def export_final(fig: plt.Figure) -> list[Path]:
    """审阅通过后导出可编辑矢量图和高分辨率位图。"""

    stem = FIGURES_DIR / "result_q1_lineage_module_state"
    outputs = [
        stem.with_suffix(".png"),
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    fig.savefig(outputs[2])
    fig.savefig(outputs[3], dpi=600)
    return outputs


def main() -> None:
    """执行选定导出模式并输出数据覆盖摘要。"""

    args = parse_args()
    figure, audit = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    print(
        "数据覆盖："
        f"{audit['kept_lineages']}个谱系、{audit['kept_cells']}个细胞；"
        f"图外说明{audit['excluded_lineages']}个谱系、{audit['excluded_cells']}个细胞。"
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
