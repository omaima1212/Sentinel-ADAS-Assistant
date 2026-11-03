"""Small, local matplotlib charts for Sentinel answers."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHART_DIR = Path(__file__).resolve().parent.parent / "results" / "charts"
ROAD_CONDITIONS = ("dry", "wet", "snow", "ice")
TIMELINE_DATA = Path(__file__).resolve().parent.parent / "data" / "timelines" / "adas_eras.json"


def stopping_distance_chart(speed_kmh, road_condition):
    """Create (or reuse) a comparison chart and return its absolute path."""
    # Use the established calculation for validation and all plotted values.
    from agent import stopping_distance

    requested = str(road_condition).lower()
    if requested not in ROAD_CONDITIONS:
        raise ValueError("Use dry, wet, snow, or ice road condition.")
    speed = float(speed_kmh)
    if speed < 0:
        raise ValueError("Speed must be nonnegative.")
    display_speed = int(speed) if speed.is_integer() else speed
    target = CHART_DIR / f"stopping_{display_speed}_{requested}.png"
    if target.exists():
        return str(target)

    values = [stopping_distance(speed, condition) for condition in ROAD_CONDITIONS]
    reaction = [item["reaction_m"] for item in values]
    braking = [item["braking_m"] for item in values]
    totals = [item["total_m"] for item in values]

    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#0B1120")
    y_positions = list(range(len(ROAD_CONDITIONS)))
    bars_reaction = ax.barh(y_positions, reaction, color="#99F6E4",
                            edgecolor="none", label="Reaction distance")
    bars_braking = ax.barh(y_positions, braking, left=reaction,
                           color="#2DD4BF", edgecolor="none",
                           label="Braking distance")
    for index, (label, total) in enumerate(zip(ROAD_CONDITIONS, totals)):
        ax.text(totals[index] + max(totals) * .015, index, f"{total:.1f} m",
                va="center", color="#F8FAFC", fontsize=9,
                fontweight="bold" if label == requested else "normal")
    ax.set_yticks(y_positions, [label.title() for label in ROAD_CONDITIONS])
    for tick, condition in zip(ax.get_yticklabels(), ROAD_CONDITIONS):
        tick.set_fontweight("bold" if condition == requested else "normal")
        tick.set_color("#F8FAFC")
    ax.invert_yaxis()
    ax.set_xlabel("Distance (m)", color="#F8FAFC")
    ax.set_title(f"Stopping distance at {display_speed} km/h",
                 color="#F8FAFC", loc="left", pad=14)
    ax.grid(axis="x", color="#273449", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#273449")
    legend = ax.legend(handles=[bars_reaction, bars_braking], frameon=False,
                       loc="lower right", fontsize=8)
    for label in legend.get_texts():
        label.set_color("#F8FAFC")
    fig.text(.01, -.01, "Simplified estimate for explanation only",
             color="#94A3B8", fontsize=8)
    ax.set_xlim(0, max(totals) * 1.25 if max(totals) else 1)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return str(target)


def adas_timeline_chart(guide_loaded=False):
    """Create (or reuse) the local ADAS evolution chart and return its path."""
    target = CHART_DIR / "adas_timeline.png"
    if target.exists():
        return str(target)

    with TIMELINE_DATA.open(encoding="utf-8") as handle:
        eras = json.load(handle)

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#0B1120")
    shades = ["#134E4A", "#115E59", "#0F766E", "#0D9488", "#2DD4BF"]
    for index, era in enumerate(eras):
        predicted = era.get("status") == "predicted"
        examples = era["examples"][:3]
        if era.get("start") == 2010 and "AEB" in era["examples"]:
            examples = ["Rear cameras", "AEB", "Rear cross-traffic alert"]
        elif era.get("start") == 2016 and "Self-park" in era["examples"]:
            examples = ["Lane keeping", "ACC", "Self-park"]
        ax.barh(0, 0.88, left=index + 0.06, height=0.64,
                color="#0B1120" if predicted else shades[index],
                edgecolor="#2DD4BF", linewidth=2,
                linestyle="--" if predicted else "-", zorder=2)
        ax.text(index + 0.5, 0.04, era["era"], ha="center", va="center",
                color="#F8FAFC", fontsize=10, fontweight="bold", wrap=True,
                zorder=3)
        ax.text(index + 0.5, -0.53, "\n".join(examples),
                ha="center", va="top", color="#F8FAFC", fontsize=8.5,
                linespacing=1.5)
        period = (f'{era["start"]}+' if era.get("end") is None else
                  f'{era["start"]}–{era["end"]}')
        ax.text(index + 0.5, 0.52, period,
                ha="center", va="bottom", color="#2DD4BF", fontsize=9,
                fontweight="bold")

    ax.annotate("", xy=(4.97, 0), xytext=(0.03, 0),
                arrowprops={"arrowstyle": "->", "color": "#2DD4BF",
                            "linewidth": 1.6}, zorder=1)
    ax.set_xlim(-0.05, 5.05)
    ax.set_ylim(-1.45, 0.95)
    ax.axis("off")
    ax.set_title("Evolution of ADAS", color="#F8FAFC", fontsize=17,
                 fontweight="bold", loc="left", pad=15)
    footer = ("Source: 2021 ADAS Guide, p. 4" if guide_loaded else
              "Based on the 2021 ADAS Guide (not currently loaded)")
    fig.text(0.01, 0.012, footer, color="#94A3B8", fontsize=8)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return str(target)
