import os

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib import font_manager as fm


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


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def mix_with_white(hex_color, mix=0.68):
    rgb = hex_to_rgb01(hex_color)
    return (1 - mix) * rgb + mix * np.ones(3)


models = ["lincal-vl-3B", "qwen2.5-vl-3B"]
DISPLAY_NAME = {
    "lincal-vl-3B": "LuxiVLM-3B",
    "qwen2.5-vl-3B": "qwen2.5-vl-3B",
}
contexts_eff = ["1K", "2K", "3K", "4K", "8K"]

speed = np.array(
    [
        [25.97, 20.44, 17.79, 12.68, 7.79],
        [27.32, 14.98, 13.41, 7.75, 4.13],
    ]
)

prefill = np.array(
    [
        [1128.46, 589.31, 445.99, 436.04, 250.34],
        [1144.31, 465.56, 305.61, 304.06, 170.45],
    ]
)

kv_cache_mib = {
    "lincal-vl-3B": [20.23, 29.23, 38.23, 47.23, 83.23],
    "qwen2.5-vl-3B": [20.23, 38.23, 56.23, 74.23, 146.23],
}
kv_cache = np.array([kv_cache_mib[model] for model in models])

THEME = {
    "bg": "#FFFFFF",
    "card": "#FFFFFF",
    "card_edge": "#E9ECF5",
    "grid": "#EEF1F7",
    "text": "#111827",
    "muted": "#6B7280",
    "accent": "#2563EB",
    "good": "#0F766E",
}

COLOR = {
    "lincal-vl-3B": "#6D5EF5",
    "qwen2.5-vl-3B": "#2BB3C0",
}
HATCH = {"lincal-vl-3B": ".."}


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


def fmt_value(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_grouped(ax, values, ylabel, *, ratio_color=None):
    add_card(ax)
    beautify_axes(ax, ylabel)

    width = 0.36
    x = np.arange(len(contexts_eff))
    vmax = values.max()

    for i, model in enumerate(models):
        off = (i - (len(models) - 1) / 2) * width
        is_qwen = model == "qwen2.5-vl-3B"

        if is_qwen:
            face = mix_with_white(COLOR[model], mix=0.68)
            edge = mix_with_white(COLOR[model], mix=0.82)
            hatch = None
            alpha = 0.95
            zorder = 2.8
        else:
            face = COLOR[model]
            edge = "#FFFFFF"
            hatch = HATCH.get(model)
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

        for bar in bars:
            if not is_qwen:
                bar.set_path_effects(
                    [
                        pe.withSimplePatchShadow(
                            offset=(0, -1.4),
                            alpha=0.18,
                            shadow_rgbFace=(0, 0, 0),
                        ),
                        pe.Normal(),
                    ]
                )
            else:
                bar.set_path_effects(
                    [
                        pe.Stroke(linewidth=3.2, foreground="white", alpha=0.65),
                        pe.Stroke(linewidth=1.2, foreground=edge, alpha=0.95),
                        pe.withSimplePatchShadow(
                            offset=(0, -0.8), alpha=0.08, shadow_rgbFace=(0, 0, 0)
                        ),
                        pe.Normal(),
                    ]
                )

        txt_color = THEME["text"] if not is_qwen else THEME["muted"]
        txt_alpha = 0.92 if not is_qwen else 0.55
        for rect, value in zip(bars, values[i]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + vmax * 0.014,
                fmt_value(float(value)),
                ha="center",
                va="bottom",
                fontsize=9,
                color=txt_color,
                alpha=txt_alpha,
                zorder=6,
            )

    for j in range(len(contexts_eff)):
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
            arrowprops=dict(
                arrowstyle="<->", color=ratio_color or THEME["accent"], lw=1.6
            ),
            zorder=7,
        )
        ax.text(
            x[j],
            y + vmax * 0.025,
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            color=ratio_color or THEME["accent"],
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
    ax.set_xticklabels(contexts_eff)
    ax.set_ylim(0, vmax * 1.28)


def legend_handles():
    return [
        Patch(
            facecolor=COLOR["lincal-vl-3B"],
            edgecolor="#FFFFFF",
            linewidth=1.0,
            hatch=HATCH.get("lincal-vl-3B"),
            label=DISPLAY_NAME["lincal-vl-3B"],
        ),
        Patch(
            facecolor=mix_with_white(COLOR["qwen2.5-vl-3B"], mix=0.68),
            edgecolor=mix_with_white(COLOR["qwen2.5-vl-3B"], mix=0.82),
            linewidth=1.0,
            label=DISPLAY_NAME["qwen2.5-vl-3B"],
        ),
    ]


def style_legend(legend):
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(THEME["card_edge"])
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(1.0)


def save_speed(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 5.8), gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor(THEME["bg"])

    axes[0].set_title("解码速度（Decode）", color=THEME["text"], pad=12)
    draw_grouped(axes[0], speed, "token/s")

    axes[1].set_title("预填充速度（Prefill）", color=THEME["text"], pad=12)
    draw_grouped(axes[1], prefill, "token/s")

    legend = fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.92),
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
        "推理速度对比（NPU Snapdragon 8Elite Gen5）",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=THEME["text"],
    )
    fig.subplots_adjust(left=0.045, right=0.98, bottom=0.11, top=0.82, wspace=0.18)
    fig.savefig(os.path.join(out_dir, "lincal_qwen_vl_speed.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "lincal_qwen_vl_speed.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_memory(out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.8))
    fig.patch.set_facecolor(THEME["bg"])
    draw_grouped(ax, kv_cache, "MiB", ratio_color=THEME["good"])

    legend = fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.835),
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
        0.93,
        "KV Cache 占用对比（NPU Snapdragon 8Elite Gen5）",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=THEME["text"],
    )
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.12, top=0.765)
    fig.savefig(os.path.join(out_dir, "lincal_qwen_vl_memory.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "lincal_qwen_vl_memory.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = "charts"
    os.makedirs(out_dir, exist_ok=True)
    save_speed(out_dir)
    save_memory(out_dir)


if __name__ == "__main__":
    main()
