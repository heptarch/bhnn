"""Regenerate every data figure in notes/figs/ from results/*.json.
Usage (repo root):  python scripts/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference categorical palette (light steps), fixed order; text and grid tokens
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#9ec5f0", "#5f9fe6", "#2a78d6", "#1a4f93"]          # one hue, light -> dark

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
OUT = "notes/figs"
os.makedirs(OUT, exist_ok=True)
R = lambda name: json.load(open(f"results/{name}.json"))


def fig_e02():
    d = R("e02_tms_vs_haar")
    mods = {(m["m"], m["p"]): m for m in d["models"]}
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    bins = np.linspace(-1, 1, 81)
    for a, key in ((ax[0], (256, 0.01)), (ax[1], (64, 0.03))):
        m = mods[key]
        a.hist(m["overlaps_trained"], bins=bins, density=True, color=S[0], alpha=0.85, label="trained")
        h, e = np.histogram(m["overlaps_haar"], bins=bins, density=True)
        a.plot(0.5 * (e[1:] + e[:-1]), h, color=S[1], lw=1.6, label="Haar frame")
        a.set_yscale("log"); a.set_ylim(1e-3, 50)
        a.set_xlabel(r"overlap $\langle w_i, w_j\rangle$ (unit columns)")
        a.set_title(f"$n=32$, $m={key[0]}$, $p={key[1]}$")
    ax[0].set_ylabel("density"); ax[0].legend(loc="upper right")
    ax[0].annotate("positive overlaps capped", xy=(0.25, 0.5), xytext=(0.3, 6), color=INK2, fontsize=8,
                   arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax[1].annotate("antipodal pairs", xy=(-0.98, 0.6), xytext=(-0.7, 8), color=INK2, fontsize=8,
                   arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    a = ax[2]
    for i, key in enumerate([(64, 0.03), (128, 0.03), (256, 0.01)]):
        rec = mods[key]["recovery"]["nonneg"]
        ks = sorted(int(k) for k in rec["trained"])
        kn = np.array(ks) / rec["dt_k"]
        a.plot(kn, [rec["trained"][str(k)] for k in ks], "-o", color=S[i], ms=4, label=f"trained $m={key[0]}$")
        a.plot(kn, [rec["haar"][str(k)] for k in ks], "--", color=S[i], lw=1.2)
    a.axvline(1, color=INK2, ls=":", lw=1)
    a.set_xlabel(r"$k / k^*$ (nonnegative $\ell_1$ threshold)"); a.set_ylabel("exact recovery rate")
    a.set_title("recovery: trained (solid) vs Haar (dashed)")
    a.legend(loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/e02_tms.pdf"); plt.close(fig)


NAMES = {"roberta": "RoBERTa", "glove": "GloVe"}


def fig_e03():
    d = R("e03_real_embeddings")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    a = ax[0]
    for i, name in enumerate(("roberta", "glove")):
        ev = np.array(d[name]["cov_eigs"])
        a.plot(np.arange(1, len(ev) + 1) / len(ev), ev, color=S[i], label=f"{NAMES[name]} ($n={d[name]['n']}$)")
    a.axhline(1, color=INK2, ls="--", lw=1, label="tight frame (Haar)")
    a.set_yscale("log"); a.set_xlabel("eigenvalue rank / $n$"); a.set_ylabel("eigenvalue / mean")
    a.set_title("covariance spectrum (centred)"); a.legend(fontsize=7)
    for a, name in ((ax[1], "roberta"), (ax[2], "glove")):
        for i, v in enumerate(("raw", "centred", "whitened")):
            ov = d[name]["versions"][v]["overlaps"]
            e = np.array(ov["edges"]); h = np.array(ov["hist"], float)
            h = h / h.sum() / np.diff(e)
            a.plot(0.5 * (e[1:] + e[:-1]), h, color=S[i], lw=1.5, label=v)
        z = np.linspace(-8, 8, 400)
        a.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), "--", color=INK2, lw=1, label="Haar / Gaussian")
        a.set_yscale("log"); a.set_ylim(1e-6, 1); a.set_xlim(-8, 8)
        a.set_xlabel(r"$\sqrt{n}\,\cos\theta$ for random pairs"); a.set_title(NAMES[name])
    a.set_ylim(1e-6, 30)
    ax[1].set_ylim(1e-6, 30)
    ax[1].set_ylabel("density"); ax[1].legend(fontsize=7, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/e03_embeddings.pdf"); plt.close(fig)


def band(a, steps, runs, key, color, label, ls="-"):
    Y = np.array([[r[key] for r in run] for run in runs])
    mu, sd = Y.mean(0), Y.std(0)
    a.plot(steps, mu, ls, color=color, label=label)
    a.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)


def fig_e10():
    d = R("e10_drift_conditions")
    conds = ["release", "pressure", "pressure+budget", "pressure+monitor"]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for a, c in zip(ax, conds):
        runs = [r["rows"] for r in d["runs"] if r["condition"] == c]
        steps = [r["step"] for r in runs[0]]
        band(a, steps, runs, "task", INK2, "receiver (task)", ls="--")
        band(a, steps, runs, "refit", S[0], "refit monitor")
        band(a, steps, runs, "frozen_recal", S[2], "recalibrated")
        band(a, steps, runs, "frozen", S[1], "frozen monitor")
        a.set_title(c); a.set_xlabel("training steps after release")
    ax[0].set_ylabel("F1 (mean ± sd, 5 seeds)"); ax[0].set_ylim(0.6, 1.0)
    ax[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/e10_drift.pdf"); plt.close(fig)


def fig_e11():
    d = R("e11_optimizers")
    opts = ["adam", "sgd", "hybrid", "hybrid_reverse"]          # large-step SGD diverges on some seeds; table only
    labels = {"adam": "Adam", "sgd": "SGD", "hybrid": "hybrid (SGD sender)",
              "hybrid_reverse": "reverse hybrid (Adam sender)", "sgd_large": "SGD, large lr"}
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    for col, dd in enumerate((12, 32)):
        for i, o in enumerate(opts):
            runs = [r["rows"] for r in d["runs"] if r["d"] == dd and r["optimizer"] == o]
            steps = [r["step"] for r in runs[0]]
            band(ax[2 * col], steps, runs, "neurons_per_feature", S[i], labels[o])
            Y = np.array([[r["refit"] - r["frozen_recal"] for r in run] for run in runs])
            ax[2 * col + 1].plot(steps, Y.mean(0), color=S[i], label=labels[o])
        ax[2 * col].set_title(f"$d={dd}$: spread of each feature")
        ax[2 * col].set_ylabel("neurons per feature")
        ax[2 * col + 1].set_title(f"$d={dd}$: direction drift")
        ax[2 * col + 1].set_ylabel("refit − recalibrated F1")
        for a in ax[2 * col: 2 * col + 2]:
            a.set_xlabel("steps after release")
    ax[0].set_ylim(0.9, 2.6); ax[2].set_ylim(0.9, 2.6)
    ax[1].set_ylim(-0.01, 0.09); ax[3].set_ylim(-0.01, 0.09)
    ax[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/e11_optimizers.pdf"); plt.close(fig)


def fig_e13():
    d = R("e13_saddle_hessian")
    fresh = {r["sigma"]: r for r in R("e13_fresh")}
    runs = sorted(d["runs"], key=lambda r: r["sigma"])
    sig = np.array([r["sigma"] for r in runs])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    a = ax[0]
    a.plot(sig, [np.mean(fresh[s]["mmse_fresh"]) for s in sig], "-o", color=S[0], ms=5, label="fresh noise")
    a.plot(sig, [np.mean(fresh[s]["mmse_batch"]) for s in sig], "--s", color=S[1], ms=5, label="training batch")
    a.set_yscale("log"); a.set_xlabel(r"channel noise $\sigma$"); a.set_ylabel(r"receiver residual error $\mathrm{mmse}$")
    a.set_title("the receiver memorizes the batch"); a.legend(fontsize=7)
    a = ax[1]
    series = (
        ("I-MMSE prediction (fresh)", [np.median(fresh[s]["immse_pred_fresh"]) for s in sig], S[0], "o"),
        ("envelope curvature", [np.median([x["schur"] for x in r["features"]]) for r in runs], S[1], "s"),
        (r"global $\lambda_{\min}$", [r["lam_global"] for r in runs], S[2], "D"),
    )
    for lab, y, c, m in series:
        y = np.array(y)
        a.plot(sig, np.abs(y), "-", color=c, lw=1, alpha=0.6)
        neg = y < 0
        a.plot(sig[neg], np.abs(y[neg]), m, color=c, ms=6, label=lab)
        a.plot(sig[~neg], np.abs(y[~neg]), m, mfc="white", mec=c, mew=1.3, ms=6)
    a.set_yscale("log"); a.set_xlabel(r"channel noise $\sigma$"); a.set_ylabel("|curvature|")
    a.set_title("filled: negative, open: positive"); a.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/e13_saddle.pdf"); plt.close(fig)


def fig_e14():
    d = R("e14_bilinear_handshake")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a = ax[0]
    for j, cval in enumerate(("1.0", "0.1")):
        for i, (o, lab) in enumerate((("gd", "gradient descent"), ("adam", "Adam"))):
            tr = np.array(d["curves"][cval][o]["traj"])
            t = np.arange(len(tr)) * 10 + 1
            a.plot(t, np.linalg.norm(tr, axis=1), color=S[i], ls="-" if j == 0 else "--",
                   label=f"{lab}, $c={cval}$")
    a.axhline(0.3, color=INK2, lw=0.8, ls=":")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 4e4)
    a.set_xlabel("steps"); a.set_ylabel(r"$\|(\varepsilon, r)\|$")
    a.set_title("leaving the handshake saddle"); a.legend(fontsize=6.5, loc="upper left")
    a = ax[1]
    sc = d["scaling"]
    lam = np.array([abs(x["lam"]) for x in sc])
    a.plot(lam, [x["t_gd"] for x in sc], "o", color=S[0], ms=5, label="gradient descent")
    a.plot(lam, [x["t_gd_pred"] for x in sc], "--", color=S[0], lw=1, label=r"$\log(1/z_0)/(\eta|\lambda_-|)$")
    a.plot(lam, [x["t_adam"] for x in sc], "s", color=S[1], ms=5, label="Adam")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$|\lambda_-|$"); a.set_ylabel("steps to escape")
    a.set_title("escape time vs curvature"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/e14_handshake.pdf"); plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["e02", "e03", "e10", "e11", "e13", "e14"]
    for w in which:
        if os.path.exists(f"results/{dict(e02='e02_tms_vs_haar', e03='e03_real_embeddings', e10='e10_drift_conditions', e11='e11_optimizers', e13='e13_saddle_hessian', e14='e14_bilinear_handshake')[w]}.json"):
            globals()[f"fig_{w}"](); print("made", w)
        else:
            print("missing results for", w)
