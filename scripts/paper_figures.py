"""The five figures of the arXiv manuscript, drawn from the result files (no number is typed in by hand).

    PYTHONPATH=scripts uv run python scripts/paper_figures.py [--out docs/paper/arxiv/figures]

One typographic system for all five: Helvetica Neue. Figure 1 is a TikZ drawing (docs/paper/arxiv/figures/
fig1_architecture.tex) compiled with LuaLaTeX so that it uses the same face; this script writes its numbers
(fig1_numbers.tex) from results/ and compiles it. Figures 2-5 are matplotlib: panel letters in bold at the top-left of
each axes, legends outside the data. Every panel names its source file in a comment next to the code that reads it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

D1 = Path("results/d1")
COL = {
    "human": "#1F7A4D",
    "kernel": "#D2551A",
    "hybrid": "#1C2226",
    "flagship": "#6D3FD1",
    "grey": "#9AA1A6",
    "light": "#D9DDE0",
    "ink": "#2B3136",
    "teal": "#1F7A4D",
}
ARM = {
    "evaluation": "evaluation",
    "long_evaluation": "long evaluation",
    "generic_norms": "generic norms",
    "partisan_norms": "partisan norms",
    "tips": "tips",
    "tips_norms": "tips + norms",
    "importance": "importance",
    "importance_norms": "importance + norms",
}
plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#5B6166",
        "axes.linewidth": 0.7,
        "xtick.color": "#5B6166",
        "ytick.color": "#5B6166",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.labelcolor": "#2B3136",
        "text.color": "#2B3136",
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    }
)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def title(ax, letter: str, text: str) -> None:
    ax.text(0, 1.05, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=COL["ink"])
    ax.annotate(
        text, xy=(0, 1.05), xycoords="axes fraction", xytext=(13, 0), textcoords="offset points", fontsize=8, va="bottom", ha="left", color=COL["ink"]
    )


def ygrid(ax) -> None:
    ax.grid(axis="y", color="#E6E9EB", lw=0.6)
    ax.set_axisbelow(True)


def legend_below(fig, handles, labels, *, ncol: int, bottom: float, y: float = 0.0) -> None:
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=1.6, columnspacing=1.4, handletextpad=0.6
    )
    fig.tight_layout(rect=(0, bottom, 1, 1), w_pad=1.6)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)


# ---- figure 1: architecture (TikZ) ----

FIG1_TEX = Path(__file__).resolve().parents[1] / "docs/paper/arxiv/figures/fig1_architecture.tex"


def count_lines(path: str | Path, needle: str | None = None) -> int:
    with open(path) as f:
        return sum(1 for line in f if needle is None or needle in line)


def fig1(out: Path) -> None:
    """Write the measured numbers the architecture diagram shows, then compile the TikZ source with LuaLaTeX."""
    live = load("results/c/20260923-113433-live/throughput.json")  # data: S1 live run
    fresh = live["predictions"] - live["cache_hits"] / 2  # the ledger counts one cache hit per answer option (two per prediction)
    full = "results/20260921-012513-socsci210-test-jev-p3-choice"  # data: kernel on the whole SocSci210 test population
    ledger = load(f"{full}/ledger.json")
    ledger = ledger[-1] if isinstance(ledger, list) else ledger
    population = count_lines(f"{full}/predictions.jsonl")
    prices = sorted([live["usd"] / fresh * 1000, ledger["usd"] / population * 1000])
    d1_states = count_lines("results/d1/20260922-213832-jev/predictions.jsonl", '"pool":"application"')  # data: D1 kernel run
    d1_anchors = count_lines("results/d1/20260922-225014-gpt-6-astra/predictions.jsonl", '"anchor":true')  # data: D1 flagship run
    d3_k1 = count_lines("results/b/20260923-022351-test-paired-gpt-6-astra/predictions.jsonl", '"anchor_rank":0')  # data: D3 anchors
    fractions = sorted([d3_k1 / population, d1_anchors / d1_states])
    execute = load("results/c/execute.json")  # data: S1 execution
    million = next(r for r in execute["throughput"] if r["tier"] == "vectorised" and r["agents"] == 1_000_000)
    cover = load("results/d2/discrepancy_model.json")["jev"]["test"]["coverage_with_parameter_uncertainty"]  # data: D2
    raw = load(D1 / "robustness.json")["raw_mae"]  # data: D1 appendix A
    gain = load("results/b/report.json")["table"]  # data: D3 test report
    macros = {
        "KPrice": f"\\${prices[0]:.2f}--{prices[1]:.2f}",
        "KLatency": f"{live['latency_ms_p50']:.0f}\\,ms",
        "KAnchors": f"{fractions[0] * 100:.1f}--{fractions[1] * 100:.1f}\\%",
        "KExec": f"{million['seconds']:.2f}\\,s",
        "KCovA": f"{cover['80%']['coverage'] * 100:.0f}\\%",
        "KCovB": f"{cover['90%']['coverage'] * 100:.0f}\\%",
        "KEffErr": f"{(1 - raw['hybrid'] / raw['kernel']) * 100:.0f}\\%",
        "KGainA": f"{gain['kernel']['captured_gain']:.2f}",
        "KGainB": f"{gain['hybrid_k3']['captured_gain']:.2f}",
    }
    lines = ["% generated by scripts/paper_figures.py from results/ -- do not edit by hand"]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    if out.resolve() != FIG1_TEX.parent:
        shutil.copy(FIG1_TEX, out / FIG1_TEX.name)
    (out / "fig1_numbers.tex").write_text("\n".join(lines) + "\n")
    run = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", FIG1_TEX.name], cwd=out, capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"lualatex failed on {FIG1_TEX.name}:\n{run.stdout[-1500:]}")
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", "fig1_architecture.pdf", "fig1_architecture"], cwd=out, check=True)
    for suffix in (".aux", ".log"):
        (out / f"fig1_architecture{suffix}").unlink(missing_ok=True)


# ---- figure 2: sparse calibration ----


def fig2(out: Path) -> None:
    robust = load(D1 / "robustness.json")  # data: results/d1/robustness.json
    report = load("results/b/report.json")  # data: results/b/report.json
    fig = plt.figure(figsize=(6.6, 5.1))
    grid = fig.add_gridspec(2, 12, height_ratios=[1, 1.02], hspace=0.55, wspace=1.2, left=0.08, right=0.985, top=0.94, bottom=0.1)
    ax_a, ax_b = fig.add_subplot(grid[0, 0:6]), fig.add_subplot(grid[0, 6:12])
    ax_c, ax_key = fig.add_subplot(grid[1, 0:8]), fig.add_subplot(grid[1, 8:12])
    floor = robust["sampling_floor"]["mae_floor_if_model_perfect"]

    # A: Epstein raw MAE with participant-bootstrap intervals, calibrated parents, floor
    ax = ax_a
    systems = [("kernel", "kernel", COL["kernel"]), ("hybrid", "hybrid", COL["hybrid"]), ("flagship_audit", "flagship audit", COL["flagship"])]
    k = robust["kernel_minus_hybrid_abs_error"]
    for i, (key, _lab, c) in enumerate(systems):
        v, ci = robust["raw_mae"][key], k["mae_participant_ci95"][key]
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=c, ms=5.5, capsize=3, lw=1)
        ax.text(i - 0.13, v, f"{v:.4f}", ha="right", va="center", fontsize=6.3, color=c)
        cal = robust["calibrated_parents"][key]["mae_loo_slope"]
        ax.plot(i + 0.24, cal, marker="D", ms=4.3, color=c, mfc="white", mew=1.1, ls="none")
    ax.axhline(floor, ls=(0, (2, 2)), color=COL["grey"], lw=0.9)
    ax.text(2.62, floor - 0.0012, "perfect-model floor", fontsize=6, color="#7A8187", va="top", ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels([s_[1] for s_ in systems])
    ax.set_xlim(-0.6, 2.65)
    ax.set_ylim(0, 0.05)
    ax.set_ylabel("effect MAE over 10 effects")
    ygrid(ax)
    title(ax, "A", "Epstein: intervention-effect error")

    # B: persona-count curve, labelled at the line ends
    ax = ax_b
    curve = robust["persona_count_curve"]
    n = [r["personas_per_wave"] for r in curve]
    for key, c, lab in (("kernel", COL["kernel"], "kernel"), ("hybrid", COL["hybrid"], "hybrid")):
        m, s_ = np.array([r[f"{key}_mae"] for r in curve]), np.array([r[f"{key}_sd"] for r in curve])
        ax.plot(n, m, marker="o", ms=3.6, color=c, lw=1.5)
        ax.fill_between(n, m - s_, m + s_, color=c, alpha=0.13, lw=0)
        ax.text(n[-1] * 1.08, m[-1], lab, va="center", ha="left", fontsize=6.8, color=c)
    ax.plot([30], [robust["raw_mae"]["flagship_audit"]], marker="^", ms=6.5, color=COL["flagship"], ls="none")
    ax.text(30 * 1.1, robust["raw_mae"]["flagship_audit"] + 0.0012, "flagship audit", fontsize=6.3, color=COL["flagship"], va="bottom")
    ax.axhline(floor, ls=(0, (2, 2)), color=COL["grey"], lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlim(8.5, 290)
    ax.set_ylim(0, 0.05)
    ax.set_xlabel("simulated personas per wave")
    ax.tick_params(axis="y", labelleft=False)  # same scale as panel A
    ygrid(ax)
    title(ax, "B", "Epstein: more personas buy no accuracy")

    # C: SocSci210 captured gain with study-bootstrap intervals
    ax = ax_c
    names = [
        ("kernel", "kernel", COL["kernel"]),
        ("flagship5", "flagship\n5 per cell", COL["flagship"]),
        ("hybrid_k1", "hybrid k = 1\n694 anchors", COL["hybrid"]),
        ("hybrid_k2", "hybrid k = 2\n1,388 anchors", COL["hybrid"]),
        ("hybrid_k3", "hybrid k = 3\n2,082 anchors", COL["hybrid"]),
    ]
    for i, (key, _lab, c) in enumerate(names):
        v, ci = report["table"][key]["captured_gain"], report["ci95"][key]["captured_gain"]
        ax.bar(i, v, color=c, alpha=0.32 if key in ("hybrid_k1", "hybrid_k2") else 0.88, width=0.58)
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=COL["ink"], capsize=3, lw=1)
        ax.text(i + 0.33, v, f"{v:.2f}", ha="left", va="center", fontsize=6.3, color="#5B6166")
    d = report["differences"]["hybrid_k3-kernel:captured_gain"]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([nm[1] for nm in names], fontsize=6.6)
    ax.set_xlim(-0.55, 4.75)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("captured decision gain")
    ygrid(ax)
    note = f"hybrid k = 3 − kernel: {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]\napproximation endpoint: {report['primary']['verdict']}"
    ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", ha="left", fontsize=6.4)
    title(ax, "C", "SocSci210: 37 unseen studies")

    # key, beside panel C
    ax_key.axis("off")
    handles = [
        Line2D([], [], marker="o", color=COL["kernel"], ls="none", ms=5.5),
        Line2D([], [], marker="o", color=COL["hybrid"], ls="none", ms=5.5),
        Line2D([], [], marker="^", color=COL["flagship"], ls="none", ms=6),
        Line2D([], [], marker="D", color=COL["ink"], mfc="white", mew=1.1, ls="none", ms=4.3),
        Line2D([], [], color=COL["grey"], ls=(0, (2, 2)), lw=0.9),
        Line2D([], [], color=COL["ink"], lw=1),
        Patch(fc=COL["ink"], alpha=0.13),
    ]
    labels = [
        "kernel (Jev)",
        "hybrid: kernel + paired\nflagship anchors",
        "flagship direct audit,\n30 personas per wave",
        "calibrated out of wave (A)",
        "perfect-model floor:\nhuman sampling (A, B)",
        "95% bootstrap interval:\nparticipants (A), studies (C)",
        "±1 sd over persona draws (B)",
    ]
    ax_key.legend(handles, labels, loc="center left", frameon=False, handlelength=1.5, handletextpad=0.7, labelspacing=0.95, borderaxespad=0)
    save(fig, out, "fig2_sparse_calibration")


# ---- figure 3: discrepancy ----


def fig3(out: Path) -> None:
    d2 = load("results/d2/discrepancy_model.json")["jev"]  # data: results/d2/discrepancy_model.json
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    mc = d2["monte_carlo_versus_discrepancy"]
    n = [r["n_per_cell"] for r in mc]
    ax.plot(n, [r["monte_carlo_sd"] for r in mc], marker="o", ms=3.3, color=COL["kernel"], lw=1.4)
    ax.axhline(mc[0]["discrepancy_sd"], color=COL["teal"], lw=1.6)
    ax.text(
        n[-1], mc[0]["discrepancy_sd"] + 0.003, f"√(τ² + σ²) = {mc[0]['discrepancy_sd']:.3f}", ha="right", va="bottom", fontsize=6, color=COL["teal"]
    )
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlabel("simulated respondents per cell")
    ax.set_ylabel("standard deviation, unit scale")
    ax.set_ylim(0, 0.1)
    ygrid(ax)
    title(ax, "A", "Only the Monte-Carlo term shrinks")

    ax = axes[1]
    levels = ["50%", "80%", "90%"]
    series = [("coverage_with_parameter_uncertainty", COL["teal"]), ("coverage_dev_fit", COL["grey"]), ("coverage_no_discrepancy", COL["kernel"])]
    w = 0.26
    for j, (key, c) in enumerate(series):
        vals = [d2["test"][key][lv]["coverage"] for lv in levels]
        ax.bar(np.arange(3) + (j - 1) * w, vals, width=w, color=c, alpha=0.88)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.02, f"{v:.0%}", ha="center", fontsize=5.6, color="#5B6166")
    for i, lv in enumerate(levels):
        ax.plot([i - 0.45, i + 0.45], [float(lv[:-1]) / 100] * 2, ls=(0, (3, 2)), color=COL["ink"], lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"nominal {lv}" for lv in levels])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("coverage of 532 human effects")
    ygrid(ax)
    title(ax, "B", "Coverage on 37 unseen studies")

    handles = [
        Line2D([], [], marker="o", color=COL["kernel"], lw=1.4, ms=3.3),
        Line2D([], [], color=COL["teal"], lw=1.6),
        Patch(fc=COL["teal"]),
        Patch(fc=COL["grey"]),
        Patch(fc=COL["kernel"]),
        Line2D([], [], color=COL["ink"], ls=(0, (3, 2)), lw=0.9),
    ]
    labels = [
        "Monte-Carlo sd of a predicted effect",
        "shared discrepancy sd",
        "interval with the discrepancy model",
        "model fit on 20 unselected dev studies",
        "human sampling error only",
        "nominal level",
    ]
    legend_below(fig, handles, labels, ncol=3, bottom=0.2)
    save(fig, out, "fig3_discrepancy")


# ---- figure 4: Epstein effects and headline heterogeneity ----


def fig4(out: Path) -> None:
    report, policies = load(D1 / "report.json"), load(D1 / "policies.json")  # data: results/d1/report.json, policies.json
    human_ci = {f"{r['wave']}:{r['arm']}": r for r in load(D1 / "human_effect_intervals.json")}  # data: results/d1/human_effect_intervals.json
    heads = load(D1 / "headlines.json")  # data: results/d1/headlines.json
    H, T = report["human"], policies["tables"]
    keys = sorted([k for k in H if not k.endswith(":control")], key=lambda k: (int(k.split(":")[0]), k.split(":")[1]))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for i, key in enumerate(keys):
        w, a = key.split(":")
        h = H[key]["discernment"] - H[f"{w}:control"]["discernment"]
        ci = human_ci[key]["ci95"]
        ax.plot([ci[0], ci[1]], [i, i], color=COL["human"], lw=4, alpha=0.22, solid_capstyle="butt")
        ax.plot(h, i, "o", color=COL["human"], ms=5.2, zorder=3)
        for sys_, marker, c, mfc in (
            ("kernel", "o", COL["kernel"], COL["kernel"]),
            ("hybrid", "s", COL["hybrid"], "white"),
            ("flagship_audit", "^", COL["flagship"], COL["flagship"]),
        ):
            e = T[sys_][key]["discernment"] - T[sys_][f"{w}:control"]["discernment"]
            ax.plot(e, i, marker, color=c, ms=4.3, mfc=mfc, mew=1.1, zorder=4)
        flag = "†" if key == "3:generic_norms" else ("‡" if key == "5:importance" else "")
        if flag:
            ax.text(0.118, i, flag, va="center", ha="left", fontsize=8, color=COL["ink"])
    ax.axvline(0, color="#B9BFC4", lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"wave {k.split(':')[0]} · {ARM[k.split(':')[1]]}" for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(-0.045, 0.13)
    ax.set_xlabel("effect on discernment vs the wave's control")
    ax.grid(axis="x", color="#E6E9EB", lw=0.6)
    ax.set_axisbelow(True)
    title(ax, "A", "Ten wave-relative effects")

    ax = axes[1]
    preds = [("kernel", COL["kernel"]), ("hybrid", COL["hybrid"]), ("flagship_audit", COL["flagship"]), ("baseline", COL["grey"])]
    for i, (key, c) in enumerate(preds):
        for stat, ck, off, alpha in (
            ("within_arm_pearson", "within_arm", -0.17, 0.88),
            ("within_arm_veracity_pearson", "within_arm_veracity", 0.17, 0.4),
        ):
            v, ci = heads["stats"][key][stat], heads["ci95"][key][ck]
            ax.bar(i + off, v, width=0.32, color=c, alpha=alpha)
            ax.errorbar(i + off, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=COL["ink"], capsize=2.2, lw=0.8)
    ax.axhline(0, color="#B9BFC4", lw=0.8)
    rel = heads["human_split_half_reliability_within_arm"]
    ax.axhline(np.sqrt(rel), ls=(0, (2, 2)), color=COL["human"], lw=0.9)
    ax.text(3.55, np.sqrt(rel) + 0.015, f"ceiling √reliability = {np.sqrt(rel):.2f}", ha="right", fontsize=5.8, color=COL["human"])
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels(["kernel", "hybrid", "flagship audit", "baseline only"], fontsize=6.3, rotation=18, ha="right")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("correlation with people's headline effects")
    ax.set_ylim(-0.42, 0.78)
    ygrid(ax)
    title(ax, "B", "Headline heterogeneity, 200 effects")

    handles = [
        Line2D([], [], marker="o", color=COL["human"], ls="none", ms=5.2),
        Line2D([], [], marker="o", color=COL["kernel"], ls="none", ms=4.3),
        Line2D([], [], marker="s", color=COL["hybrid"], mfc="white", mew=1.1, ls="none", ms=4.3),
        Line2D([], [], marker="^", color=COL["flagship"], ls="none", ms=4.6),
        Patch(fc=COL["ink"], alpha=0.88),
        Patch(fc=COL["ink"], alpha=0.4),
    ]
    labels = ["people (95% participant bootstrap)", "kernel", "hybrid", "flagship audit", "centred within arm", "centred within arm × veracity"]
    fig.text(
        0.01,
        0.005,
        "† the kernel's choice in wave 3, people's worst arm    ‡ the hybrid followed the flagship in wave 5",
        fontsize=5.8,
        color="#7A8187",
        va="bottom",
    )
    legend_below(fig, handles, labels, ncol=3, bottom=0.2, y=0.035)
    save(fig, out, "fig4_epstein_effects")


# ---- figure 5: scale ----


def fig5(out: Path) -> None:
    ex = load("results/c/execute.json")  # data: results/c/execute.json
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    rows = ex["throughput"]
    sup = {4: "10⁴", 5: "10⁵", 6: "10⁶"}
    labels = [f"{sup[int(round(np.log10(r['agents'])))]} agents" + ("\nscalar tier" if r["tier"] == "scalar" else "") for r in rows]
    colors = [COL["grey"] if r["tier"] == "scalar" else COL["teal"] for r in rows]
    ax.bar(range(len(rows)), [r["agent_steps_per_second"] for r in rows], color=colors, width=0.62)
    for i, r in enumerate(rows):
        label = f"{r['seconds'] * 1000:.0f} ms" if r["seconds"] < 1 else f"{r['seconds']:.2f} s"
        ax.text(i, r["agent_steps_per_second"] * 1.28, label, ha="center", fontsize=6.3, color="#5B6166")
    ax.set_yscale("log")
    ax.set_ylim(1e5, 1e8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("agent-steps per second")
    ygrid(ax)
    title(ax, "A", "Execution from the table, 20 steps per agent")

    ax = axes[1]
    crn = ex["crn"]
    ax.barh([0, 1], [crn["effect_sd_independent"], crn["effect_sd_crn"]], color=[COL["grey"], COL["teal"]], height=0.58)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["independent draws", "common random\nnumbers"])
    for i, v in enumerate([crn["effect_sd_independent"], crn["effect_sd_crn"]]):
        ax.text(v + 0.00008, i, f"{v:.5f}", va="center", fontsize=6.3, color="#5B6166")
    ax.set_xlim(0, crn["effect_sd_independent"] * 1.45)
    ax.set_xlabel(f"sd of the treatment effect, {crn['seeds']} seeds, {crn['agents']:,} agents")
    ax.grid(axis="x", color="#E6E9EB", lw=0.6)
    ax.set_axisbelow(True)
    title(ax, "B", f"Common random numbers: variance ÷ {(crn['effect_sd_independent'] / crn['effect_sd_crn']) ** 2:.0f}")
    handles = [Patch(fc=COL["teal"]), Patch(fc=COL["grey"])]
    legend_below(
        fig,
        handles,
        ["vectorized tier (SplitMix64 event keys, inverse CDF)", "scalar tier (blake2b event keys, one agent at a time)"],
        ncol=2,
        bottom=0.16,
    )
    save(fig, out, "fig5_scale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/paper/arxiv/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(args.out)
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
