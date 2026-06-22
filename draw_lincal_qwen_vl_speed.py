import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Patch


def set_chinese_font():
    font_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                fm.fontManager.addfont(font_file)
            except Exception:
                pass

    preferred = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name] + [
                candidate for candidate in preferred if candidate != name
            ]
            return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


set_chinese_font()

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

CONTEXTS = ["1k", "2k", "4k", "8k"]
PARAM_GROUPS = {
    "0.8B": ["lincal3.5-0.8B", "qwen3.5-0.8B"],
    "2B": ["lincal3.5-2B", "qwen3.5-2B"],
}
DISPLAY_NAME = {
    "lincal3.5-0.8B": "LinCal3.5-0.8B",
    "lincal3.5-2B": "LinCal3.5-2B",
    "qwen3.5-0.8B": "Qwen3.5-0.8B",
    "qwen3.5-2B": "Qwen3.5-2B",
}

DATA = {
    "decode": {
        "qwen3.5-0.8B": [78.747, 71.801, 64.074, 53.4],
        "qwen3.5-2B": [37.407, 35.939, 34.616, 32.7],
        "lincal3.5-0.8B": [86.994, 86.956, 86.971, 86.654],
        "lincal3.5-2B": [37.578, 36.859, 37.387, 37.133],
    },
    "prefill": {
        "qwen3.5-0.8B": [599.331, 567.544, 503.841, 409.0],
        "qwen3.5-2B": [488.816, 465.473, 454.267, 446.0],
        "lincal3.5-0.8B": [592.595, 598.131, 603.134, 603.537],
        "lincal3.5-2B": [494.756, 498.684, 508.035, 511.027],
    },
    "cache": {
        "qwen3.5-0.8B": [46.69, 70.69, 118.69, 214.69],
        "qwen3.5-2B": [46.69, 70.69, 118.69, 214.69],
        "lincal3.5-0.8B": [34.69, 34.69, 34.69, 34.69],
        "lincal3.5-2B": [34.69, 34.69, 34.69, 34.69],
    },
}

METRICS = [
    ("decode", "解码速度（Decode）", "token/s"),
    ("prefill", "预填充速度（Prefill）", "token/s"),
    ("cache", "Cache/State 占用", "MiB"),
]

THEME = {
    "bg": "#FFFFFF",
    "card": "#FFFFFF",
    "card_edge": "#E9ECF5",
    "grid": "#EEF1F7",
    "text": "#111827",
    "muted": "#6B7280",
    "accent": "#2563EB",
}

COLOR = {
    "lincal": "#6D5EF5",
    "qwen": "#D1D5DB",
}
QWEN_EDGE = "#F3F4F6"
HATCH = {"lincal": ".."}


def add_card(ax):
    ax.set_facecolor("none")
    pad_x, pad_y = 0.03, 0.08
    card = FancyBboxPatch(
        (-pad_x, -pad_y),
        1 + 2 * pad_x,
        1 + 2 * pad_y,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        transform=ax.transAxes,
        facecolor=THEME["card"],
        edgecolor=THEME["card_edge"],
        linewidth=1.2,
        zorder=-2,
        clip_on=False,
    )
    card.set_path_effects(
        [
            pe.withSimplePatchShadow(
                offset=(0, -2), alpha=0.16, shadow_rgbFace=(0, 0, 0)
            ),
            pe.Normal(),
        ]
    )
    ax.add_patch(card)


def beautify_axes(ax, ylabel):
    ax.set_ylabel(ylabel, color=THEME["muted"])
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=1.0, color=THEME["grid"])
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", colors=THEME["muted"], length=0)
    ax.tick_params(axis="y", colors=THEME["muted"], length=0)


def model_kind(model):
    return "qwen" if model.startswith("qwen") else "lincal"


def draw_grouped(ax, models, values, ylabel):
    add_card(ax)
    beautify_axes(ax, ylabel)

    width = 0.36
    x = np.arange(len(CONTEXTS))
    vmax = values.max()

    for i, model in enumerate(models):
        off = (i - (len(models) - 1) / 2) * width
        is_qwen = model_kind(model) == "qwen"

        if is_qwen:
            face = COLOR["qwen"]
            edge = QWEN_EDGE
            hatch = None
            alpha = 0.70
            zorder = 2.75
        else:
            face = COLOR["lincal"]
            edge = "#FFFFFF"
            hatch = HATCH["lincal"]
            alpha = 0.92
            zorder = 3.0

        bars = ax.bar(
            x + off,
            values[i],
            width=width,
            color=face,
            edgecolor=edge,
            linewidth=1.2 if not is_qwen else 1.0,
            alpha=alpha,
            hatch=hatch,
            zorder=zorder,
        )

        for rect in bars:
            if is_qwen:
                rect.set_path_effects(
                    [
                        pe.Stroke(linewidth=3.2, foreground="white", alpha=0.80),
                        pe.Stroke(linewidth=1.1, foreground=QWEN_EDGE, alpha=1.0),
                        pe.withSimplePatchShadow(
                            offset=(0, -0.8),
                            alpha=0.05,
                            shadow_rgbFace=(0, 0, 0),
                        ),
                        pe.Normal(),
                    ]
                )
            else:
                rect.set_path_effects(
                    [
                        pe.withSimplePatchShadow(
                            offset=(0, -1.4),
                            alpha=0.18,
                            shadow_rgbFace=(0, 0, 0),
                        ),
                        pe.Normal(),
                    ]
                )

        txt_color = THEME["muted"] if is_qwen else THEME["text"]
        txt_alpha = 0.70 if is_qwen else 0.92
        for rect, value in zip(bars, values[i]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + vmax * 0.014,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=txt_color,
                alpha=txt_alpha,
                zorder=6,
            )

    for j in range(len(CONTEXTS)):
        v0, v1 = values[0][j], values[1][j]
        if v0 >= v1:
            ratio = v0 / v1
            higher_off, lower_off = -width / 2, width / 2
        else:
            ratio = v1 / v0
            higher_off, lower_off = width / 2, -width / 2

        y = max(v0, v1) + vmax * 0.14
        ax.annotate(
            "",
            xy=(x[j] + lower_off, y),
            xytext=(x[j] + higher_off, y),
            arrowprops=dict(arrowstyle="<->", color=THEME["accent"], lw=1.6),
            zorder=7,
        )
        ax.text(
            x[j],
            y + vmax * 0.025,
            f"{ratio:.2f}×",
            ha="center",
            va="bottom",
            fontsize=10,
            color=THEME["accent"],
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.18",
                facecolor="white",
                edgecolor=THEME["card_edge"],
                linewidth=1.0,
                alpha=0.98,
            ),
            zorder=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONTEXTS)
    ax.set_ylim(0, vmax * 1.28)


def legend_handles(models):
    handles = []
    for model in models:
        if model_kind(model) == "qwen":
            handles.append(
                Patch(
                    facecolor=COLOR["qwen"],
                    edgecolor=QWEN_EDGE,
                    linewidth=1.0,
                    label=DISPLAY_NAME[model],
                )
            )
        else:
            handles.append(
                Patch(
                    facecolor=COLOR["lincal"],
                    edgecolor="#FFFFFF",
                    linewidth=1.0,
                    hatch=HATCH["lincal"],
                    label=DISPLAY_NAME[model],
                )
            )
    return handles


def style_legend(legend):
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(THEME["card_edge"])
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(1.0)


def save_param_group(out_dir, param_size, models):
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.7), gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor(THEME["bg"])

    for ax, (key, title, ylabel) in zip(axes, METRICS):
        values = np.array([DATA[key][model] for model in models], dtype=float)
        ax.set_title(title, color=THEME["text"], pad=12)
        draw_grouped(ax, models, values, ylabel)

    legend = fig.legend(
        handles=legend_handles(models),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.925),
        frameon=True,
        fancybox=True,
        borderpad=0.55,
        columnspacing=1.6,
        handlelength=1.6,
        fontsize=11,
    )
    style_legend(legend)

    fig.text(
        0.5,
        0.995,
        "推理速度与内存占用对比（NPU: Snapdragon 8Elite Gen5）",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=THEME["text"],
    )
    fig.text(
        0.145,
        0.948,
        "速度越高越好 ↑ / Cache越低越好 ↓",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=THEME["accent"],
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.25",
            facecolor="white",
            edgecolor=THEME["accent"],
            linewidth=1.2,
            alpha=0.98,
        ),
    )

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.11, top=0.78, wspace=0.18)

    safe_param_size = param_size.replace(".", "_").lower()
    fig.savefig(
        os.path.join(out_dir, f"lincal_qwen_{safe_param_size}_compare.png"),
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(out_dir, f"lincal_qwen_{safe_param_size}_compare.pdf"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    out_dir = "charts"
    os.makedirs(out_dir, exist_ok=True)
    for param_size, models in PARAM_GROUPS.items():
        save_param_group(out_dir, param_size, models)


if __name__ == "__main__":
    main()
