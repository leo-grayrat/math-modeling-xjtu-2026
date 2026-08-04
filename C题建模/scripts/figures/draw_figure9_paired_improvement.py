#!/usr/bin/env python3
"""绘制图9：两步核岭相对基础分模型的配对指标变化。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_PATH = RESULTS_DIR / "stage7_paired_test_summary.csv"
BOOTSTRAP_PATH = RESULTS_DIR / "stage7_paired_bootstrap_relative_changes.csv"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import PALETTE, apply_publication_style, audit_design, audit_layout


METRIC_ORDER = [
    "official_final_score",
    "ndcg_score",
    "precision_score",
    "rmse_reduction",
    "rho_macro",
]
METRIC_LABELS = {
    "official_final_score": "综合评分",
    "ndcg_score": "nDCG",
    "precision_score": "Precision",
    "rmse_reduction": "RMSE",
    "rho_macro": "Spearman",
}


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
    return FontProperties(fname=zh_path), FontProperties(fname=en_path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取配对检验汇总和逐次 Bootstrap 相对变化。"""

    data = pd.read_csv(DATA_PATH)
    required = {
        "metric",
        "baseline_value",
        "observed_improvement",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_repeats",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"图9数据缺少字段：{sorted(required - set(data.columns))}")
    if set(data["metric"]) != set(METRIC_ORDER) or len(data) != len(METRIC_ORDER):
        raise ValueError("图9数据必须恰好包含五项固定指标")
    if data.duplicated("metric").any():
        raise ValueError("图9数据存在重复指标")
    numeric_columns = [
        "baseline_value",
        "observed_improvement",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_repeats",
    ]
    if not np.isfinite(data[numeric_columns]).all().all():
        raise ValueError("图9数据中存在缺失或无穷数值")
    if (data["baseline_value"] <= 0).any():
        raise ValueError("相对变化的基础值必须为正")
    if not (
        (data["bootstrap_ci_lower"] <= data["observed_improvement"])
        & (data["observed_improvement"] <= data["bootstrap_ci_upper"])
    ).all():
        raise ValueError("实际变化必须位于对应 Bootstrap 区间内")
    if not (data["bootstrap_repeats"] == 10000).all():
        raise ValueError("图9契约要求所有指标均使用10000次 Bootstrap")

    data = data.set_index("metric").loc[METRIC_ORDER].reset_index()
    for source, target in [
        ("observed_improvement", "relative_change"),
        ("bootstrap_ci_lower", "relative_lower"),
        ("bootstrap_ci_upper", "relative_upper"),
    ]:
        data[target] = data[source] / data["baseline_value"] * 100
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)
    bootstrap_required = {"bootstrap_repeat", *METRIC_ORDER}
    if not bootstrap_required.issubset(bootstrap.columns):
        raise ValueError(
            f"图9 Bootstrap 数据缺少字段："
            f"{sorted(bootstrap_required - set(bootstrap.columns))}"
        )
    if len(bootstrap) != 10000 or bootstrap["bootstrap_repeat"].nunique() != 10000:
        raise ValueError("图9必须包含10000次不重复的 Bootstrap 结果")
    if not np.isfinite(bootstrap[METRIC_ORDER]).all().all():
        raise ValueError("图9 Bootstrap 数据中存在缺失或无穷数值")
    for row in data.itertuples(index=False):
        samples = bootstrap[row.metric].to_numpy(dtype=float)
        calculated = np.quantile(samples, [0.025, 0.975])
        expected = np.array([row.relative_lower, row.relative_upper])
        if not np.allclose(calculated, expected, atol=1e-3):
            raise ValueError(f"{row.metric} 的逐次分布与汇总区间不一致")
    return data, bootstrap


def draw_figure() -> tuple[plt.Figure, pd.DataFrame]:
    """按照配对改进区间契约绘制图9。"""

    apply_publication_style(language="zh", width="single")
    mpl.rcParams["figure.constrained_layout.use"] = False
    zh_font, en_font = configure_fonts()
    data, bootstrap = load_data()

    figure, axis = plt.subplots(figsize=(3.5, 2.55), layout=None)
    figure.subplots_adjust(left=0.225, right=0.975, bottom=0.235, top=0.95)

    y_positions = np.arange(len(data))[::-1]
    distributions = [bootstrap[metric].to_numpy(dtype=float) for metric in METRIC_ORDER]
    violin_parts = axis.violinplot(
        distributions,
        positions=y_positions,
        vert=False,
        widths=0.64,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        points=160,
        bw_method=0.25,
    )
    for body, row in zip(violin_parts["bodies"], data.itertuples(index=False)):
        color = PALETTE["positive"] if row.relative_change >= 0 else PALETTE["risk"]
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.72)
        body.set_zorder(2)

    for y_position, row, samples in zip(
        y_positions,
        data.itertuples(index=False),
        distributions,
    ):
        estimate = float(row.relative_change)
        lower = float(row.relative_lower)
        upper = float(row.relative_upper)
        point_color = "#238E62" if estimate >= 0 else "#E87356"
        axis.hlines(
            y_position,
            lower,
            upper,
            color=point_color,
            linewidth=1.0,
            zorder=3,
        )
        axis.vlines(
            [lower, upper],
            y_position - 0.075,
            y_position + 0.075,
            color=point_color,
            linewidth=0.85,
            zorder=3,
        )
        axis.scatter(
            estimate,
            y_position,
            marker="D",
            s=25,
            facecolor=point_color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        label = f"{estimate:+.2f}%".replace("-", "−")
        text_x = max(upper, float(np.quantile(samples, 0.995))) + 0.08
        horizontal_alignment = "left"
        axis.text(
            text_x,
            y_position,
            label,
            ha=horizontal_alignment,
            va="center",
            fontproperties=en_font,
            fontsize=6.6,
            color=PALETTE["dark"],
        )

    axis.axvline(0, color=PALETTE["dark"], linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
    axis.set_xlim(-1.55, 5.55)
    axis.set_xticks([-1, 0, 1, 2, 3, 4, 5])
    axis.set_xlabel("相对基础分模型的变化（%）", fontproperties=zh_font, fontsize=7.3, labelpad=6)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [METRIC_LABELS[metric] for metric in METRIC_ORDER],
        fontproperties=zh_font,
        fontsize=7.0,
    )
    axis.tick_params(axis="y", length=0, pad=8)
    axis.tick_params(axis="x", length=3, width=0.6, pad=4)
    for label in axis.get_xticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(6.5)
    axis.grid(axis="x", color="#E7EBE8", linewidth=0.6, zorder=0)
    axis.spines["bottom"].set_color(PALETTE["dark"])
    axis.spines["bottom"].set_linewidth(0.7)
    axis.spines["left"].set_visible(False)

    layout_issues = audit_layout(figure)
    design_issues = audit_design(figure)
    if layout_issues or design_issues:
        raise ValueError("图9预检未通过：" + "；".join(layout_issues + design_issues))
    return figure, data


def export_preview(figure: plt.Figure) -> Path:
    """只导出一张供交互审阅的 PNG。"""

    output = FIGURES_DIR / "result_q2_paired_improvement_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    return output


def export_final(figure: plt.Figure) -> list[Path]:
    """审阅通过后导出论文使用的完整格式。"""

    stem = FIGURES_DIR / "result_q2_paired_improvement"
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
    figure, _ = draw_figure()
    outputs = [export_preview(figure)] if args.preview else export_final(figure)
    plt.close(figure)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
