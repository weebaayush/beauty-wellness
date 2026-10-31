"""Week 6 - communication visuals: stakeholder dashboard mock-up, recommendation priority matrix, roadmap.

    python -m src.evaluation.communication_figures     (run after src.evaluation.evaluate)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "week6" / "figures"
INK, MUTED, GRID, ACCENT, GREY, PALE = "#1a1a1a", "#6b6b6b", "#e6e6e6", "#E8590C", "#9a9a9a", "#FFF1E8"
plt.rcParams.update({"font.family": "DejaVu Sans"})
E = json.load(open(ROOT / "reports" / "evaluation.json"))
W5 = json.load(open(ROOT / "reports" / "robustness.json"))
W4 = json.load(open(ROOT / "reports" / "eda_results.json"))


def box(ax, x, y, w, h, fc="white", ec=GRID):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=1))


# ---------------- 1. dashboard mock-up
fig = plt.figure(figsize=(7.5, 4.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, .01, .01, .98, .98, fc="#FAFAFA")
ax.text(.03, .94, "Skin-Fit Insights  ·  Monthly dashboard (mock-up)", fontsize=11, fontweight="bold", color=INK)
ax.text(.97, .94, "Filters:  Category ▾   Skin type ▾   Brand ▾", fontsize=7.5, color=MUTED, ha="right")
kpis = [("Model AUC", f"{E['overall']['auc']:.2f}", "target ≥ 0.72  ✔"),
        ("Unhappy shoppers caught", f"{E['rates']['recall']*100:.0f}%", "of all unhappy shoppers"),
        ("Saving / 1,000 shoppers", f"₹{(W5['threshold']['flag_all_per_1000']-W5['threshold']['test_cost_per_1000'])/1000:.1f}k", "vs best simple rule"),
        ("Fairness alerts", f"{sum(v['status']=='flag' for v in E['fairness_vs_rest'].values())}", "skin tone: tan  ⚠")]
for i, (t, v, s) in enumerate(kpis):
    x = .03 + i * .24
    box(ax, x, .70, .22, .19, fc="white")
    ax.text(x + .015, .855, t, fontsize=7.5, color=MUTED)
    ax.text(x + .015, .755, v, fontsize=18, fontweight="bold", color=ACCENT if i == 3 else INK)
    ax.text(x + .015, .715, s, fontsize=6.5, color=MUTED)
# chart tile: recommendation by skin fit
box(ax, .03, .06, .45, .6, fc="white")
ax.text(.045, .62, "Recommend rate by skin fit", fontsize=8.5, fontweight="bold", color=INK)
fit = W4["skin_fit"]; labels = ["Matches claim", "Other / neutral", "No claim or unknown skin", "Opposite skin type"]
for j, k in enumerate(labels):
    yy = .52 - j * .11; v = fit[k]["rate"]
    ax.text(.045, yy + .035, k, fontsize=6.8, color=MUTED)
    ax.add_patch(plt.Rectangle((.045, yy), .38 * v, .028, color=ACCENT if j == 0 else (INK if j == 3 else GREY)))
    ax.text(.05 + .38 * v, yy + .004, f"{v*100:.0f}%", fontsize=7, color=INK)
# table tile: watch list
box(ax, .51, .06, .46, .6, fc="white")
ax.text(.525, .62, "Watch list: products trending down (example rows)", fontsize=8.5, fontweight="bold", color=INK)
rows = [("Product", "Momentum", "Risk", "Action"), ("Moisturiser A", "−0.42★", "High", "Check batch"),
        ("Serum B", "−0.31★", "High", "Review claims"), ("Cleanser C", "−0.18★", "Med", "Monitor"), ("SPF D", "−0.12★", "Med", "Monitor")]
for r, row in enumerate(rows):
    yy = .54 - r * .09
    for c, (cell, cx) in enumerate(zip(row, [.525, .68, .79, .86])):
        ax.text(cx, yy, cell, fontsize=7 if r else 7.2, fontweight="bold" if r == 0 else None,
                color=(ACCENT if (c == 2 and cell == "High") else INK) if r else MUTED)
    ax.plot([.525, .955], [yy - .025, yy - .025], color=GRID, lw=.8)
ax.text(.525, .085, "Rows are illustrative; the live version lists real products.", fontsize=6.3, color=MUTED, style="italic")
fig.savefig(FIG / "g09_dashboard_mockup.png", dpi=200, facecolor="white"); plt.close(fig)

# ---------------- 2. recommendation priority matrix
recs = [("R1", "Skin-fit ranking & badges", 4.5, 2.0), ("R2", "Cart 'better match' nudge", 4.0, 2.7), ("R3", "Collect skin type at sign-up", 3.5, 1.0),
        ("R4", "Fragrance-free variants for dry skin", 3.5, 4.0), ("R5", "Product early-warning watch list", 3.1, 1.5),
        ("R6", "Show balanced / critical reviews", 2.5, 1.0), ("R7", "Value-led price positioning", 2.9, 2.1),
        ("R8", "Close fairness gap before launch", 4.1, 1.9)]
fig, ax = plt.subplots(figsize=(7.5, 3.8), gridspec_kw={})
ax.axhline(3, color=GRID, lw=1); ax.axvline(2.5, color=GRID, lw=1)
for code, name, imp, eff in recs:
    q = "quick" if imp >= 3 and eff <= 2.5 else ("major" if imp >= 3 else "fill")
    ax.scatter(eff, imp, s=230, color=ACCENT if q == "quick" else (INK if q == "major" else GREY), edgecolor="white", lw=1.5, zorder=3)
    ax.text(eff, imp, code[1:], ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=4)
ax.text(.6, 4.82, "QUICK WINS", fontsize=8, color=ACCENT, fontweight="bold"); ax.text(2.6, 4.82, "MAJOR PROJECTS", fontsize=8, color=INK, fontweight="bold")
ax.text(.6, 2.08, "NICE TO HAVE", fontsize=8, color=MUTED, fontweight="bold")
ax.set_xlim(.5, 4.6); ax.set_ylim(2, 5); ax.set_xlabel("Effort / cost  →"); ax.set_ylabel("Expected business impact  →")
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
for i, (code, name, imp, eff) in enumerate(recs):
    fig.text(.70, .86 - i * .085, f"{code}  {name}", fontsize=7.6, color=INK)
fig.subplots_adjust(left=.08, right=.67, bottom=.13, top=.9)
ax.set_title("Recommendation priority matrix (scores are judgement-based)", loc="left", fontweight="bold", fontsize=10.5)
fig.savefig(FIG / "g10_priority_matrix.png", dpi=200, facecolor="white"); plt.close(fig)

# ---------------- 3. roadmap
tasks = [("Re-run on real Sephora data", 0, 1, INK), ("Fix fairness gap (tan, fair)", 0, 2, ACCENT), ("Collect skin type at sign-up", 1, 2, GREY),
         ("Watch list for category managers", 1, 2, GREY), ("A/B test skin-fit ranking", 2, 3, ACCENT), ("Cart nudge pilot (one category)", 3, 2, ACCENT),
         ("Full rollout + monthly retraining", 5, 3, INK), ("R&D: fragrance-free dry-skin variants", 3, 5, GREY)]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for i, (t, s, d, c) in enumerate(tasks):
    ax.barh(i, d, left=s, color=c, height=.55)
ax.set_yticks(range(len(tasks)), [t[0] for t in tasks], fontsize=7.8); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.set_xlim(0, 8); ax.set_xticks(range(0, 9))
ax.set_xticklabels([f"M{m}" for m in range(0, 9)]); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
for s_ in ["top", "right", "left"]:
    ax.spines[s_].set_visible(False)
ax.set_title("Implementation roadmap (months)", loc="left", fontweight="bold", fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "g11_roadmap.png", dpi=200, facecolor="white"); plt.close(fig)
print("communication figures written")
