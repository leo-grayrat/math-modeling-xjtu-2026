#!/usr/bin/env python3
"""绘制图6：基因基础分与双侧相似性修正的两步核岭结构。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib as mpl

# 终端环境中不打开交互窗口。
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import PALETTE, audit_design, audit_layout, apply_publication_style


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


def draw_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
    color: str,
    zh_font: FontProperties,
    en_font: FontProperties,
    title_size: float = 8.0,
    line_size: float = 6.5,
) -> None:
    """绘制带短标签的结构框。"""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, alpha=0.18),
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 0.040,
        title,
        ha="center",
        va="center",
        fontproperties=zh_font if any("\u4e00" <= char <= "\u9fff" for char in title) else en_font,
        fontsize=title_size,
        color=PALETTE["dark"],
        zorder=3,
    )
    if lines:
        axis.text(
            x + width / 2,
            y + 0.060,
            "\n".join(lines),
            ha="center",
            va="center",
            fontproperties=zh_font,
            fontsize=line_size,
            linespacing=1.35,
            color=PALETTE["dark"],
            zorder=3,
        )


def draw_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = PALETTE["dark"],
    rad: float = 0.0,
    linewidth: float = 0.9,
) -> None:
    """绘制带箭头的单向计算流程。"""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
        zorder=1,
    )
    axis.add_patch(arrow)


def draw_figure() -> plt.Figure:
    """按图6契约绘制两步核岭结构示意图。"""

    apply_publication_style(language="zh", width="double")
    mpl.rcParams["figure.constrained_layout.use"] = False
    zh_font, en_font, _ = configure_fonts()

    figure, axis = plt.subplots(figsize=(7.2, 2.85), layout=None)
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.05, top=0.98)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    input_color = PALETTE["neutral"]
    base_color = PALETTE["new"]
    similarity_color = PALETTE["contrast"]
    correction_color = PALETTE["positive"]
    approximation_color = PALETTE["risk"]

    draw_box(
        axis,
        0.02,
        0.63,
        0.145,
        0.22,
        "细胞资料",
        ("表达 · 通路", "类型 · 模块"),
        input_color,
        zh_font,
        en_font,
    )
    draw_box(
        axis,
        0.02,
        0.15,
        0.145,
        0.22,
        "基因资料",
        ("数值 · 类别", "通路 · 证据"),
        input_color,
        zh_font,
        en_font,
    )
    draw_box(
        axis,
        0.23,
        0.66,
        0.205,
        0.22,
        "基因基础分  $b_g$",
        ("已见：标签均值", "新基因：元数据岭"),
        base_color,
        zh_font,
        en_font,
    )
    draw_box(
        axis,
        0.23,
        0.27,
        0.205,
        0.23,
        "残差矩阵",
        ("训练标签 - 基础分", "未观测组合：平均残差近似"),
        approximation_color,
        zh_font,
        en_font,
        line_size=6.0,
    )
    draw_box(
        axis,
        0.52,
        0.66,
        0.17,
        0.18,
        "细胞相似度  $K_c$",
        ("相似细胞传递信息",),
        similarity_color,
        zh_font,
        en_font,
        line_size=6.2,
    )
    draw_box(
        axis,
        0.52,
        0.28,
        0.17,
        0.18,
        "基因相似度  $K_g$",
        ("相似基因传递信息",),
        similarity_color,
        zh_font,
        en_font,
        line_size=6.2,
    )
    draw_box(
        axis,
        0.73,
        0.47,
        0.12,
        0.16,
        "修正  $r_{cg}$",
        ("两侧核传递",),
        correction_color,
        zh_font,
        en_font,
        line_size=6.2,
    )
    draw_box(
        axis,
        0.88,
        0.44,
        0.105,
        0.22,
        "最终预测",
        ("$\\hat y_{cg}$", "= $b_g+r_{cg}$"),
        correction_color,
        zh_font,
        en_font,
        title_size=7.4,
        line_size=7.0,
    )

    draw_arrow(axis, (0.165, 0.74), (0.23, 0.77), color=PALETTE["dark"])
    draw_arrow(axis, (0.165, 0.25), (0.23, 0.75), color=PALETTE["dark"], rad=-0.16)
    draw_arrow(axis, (0.165, 0.25), (0.52, 0.37), color=PALETTE["dark"], rad=-0.10)
    draw_arrow(axis, (0.165, 0.83), (0.52, 0.84), color=PALETTE["dark"], rad=0.28)
    draw_arrow(axis, (0.332, 0.66), (0.332, 0.50), color=base_color)
    draw_arrow(axis, (0.435, 0.38), (0.73, 0.53), color=approximation_color, rad=0.04)
    draw_arrow(axis, (0.69, 0.75), (0.73, 0.55), color=similarity_color, rad=-0.05)
    draw_arrow(axis, (0.69, 0.37), (0.73, 0.52), color=similarity_color, rad=0.05)
    draw_arrow(axis, (0.79, 0.55), (0.88, 0.55), color=correction_color)
    draw_arrow(axis, (0.435, 0.87), (0.88, 0.64), color=base_color, rad=0.28)

    axis.text(
        0.006,
        0.93,
        "输入",
        ha="left",
        va="center",
        fontproperties=zh_font,
        fontsize=7.5,
        color=PALETTE["dark"],
    )
    axis.text(
        0.232,
        0.93,
        "基础分与残差",
        ha="left",
        va="center",
        fontproperties=zh_font,
        fontsize=7.5,
        color=PALETTE["dark"],
    )
    axis.text(
        0.522,
        0.93,
        "双侧相似性",
        ha="left",
        va="center",
        fontproperties=zh_font,
        fontsize=7.5,
        color=PALETTE["dark"],
    )
    axis.text(
        0.883,
        0.93,
        "输出",
        ha="left",
        va="center",
        fontproperties=zh_font,
        fontsize=7.5,
        color=PALETTE["dark"],
    )

    layout_issues = audit_layout(figure)
    design_issues = audit_design(figure)
    if layout_issues or design_issues:
        raise ValueError("图6预检未通过：" + "；".join(layout_issues + design_issues))
    return figure


def export_preview(figure: plt.Figure) -> Path:
    """只导出一张供交互审阅的 PNG。"""

    output = FIGURES_DIR / "result_q2_model_structure_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    return output


def export_final(figure: plt.Figure) -> list[Path]:
    """审阅通过后导出论文使用的完整格式。"""

    stem = FIGURES_DIR / "result_q2_model_structure"
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
    from PIL import Image

    with Image.open(png_path) as color_image:
        color_image.convert("RGB").convert("L").save(grayscale_path)
    return [png_path, svg_path, pdf_path, tiff_path, grayscale_path]


def main() -> None:
    """执行选定导出模式。"""

    args = parse_args()
    figure = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
