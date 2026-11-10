"""Regenerate every LaTeX table in notes/tables/ from results/*.json.
Usage (repo root):  python scripts/make_tables.py
"""
import json, os
import numpy as np

OUT = "notes/tables"
os.makedirs(OUT, exist_ok=True)
R = lambda name: json.load(open(f"results/{name}.json"))
ms = lambda v: f"{np.mean(v):.3f}\\,$\\pm$\\,{np.std(v):.3f}"


def write(name, body):
    open(f"{OUT}/{name}.tex", "w").write(body)
    print("wrote", name)


def t_e01():
    d = R("e01_haar_moments")
    rows = []
    for r in d["overlaps"]:
        rows.append(f"{r['M']} & {r['N']} & {r['c']:.2f} & {r['complex_var']:.4f} & {r['complex_pred']:.4f} & "
                    f"{r['real_var']:.4f} & {r['real_pred']:.4f} & {r['gauss_var']:.4f} & {r['gauss_pred']:.4f} & "
                    f"{r['real_kurt']:.2f} & {r['complex_abs_kurt']:.2f} \\\\")
    lub = "; ".join(f"$d_A={x['dA']}$: {x['mc']:.4f} ({x['pred']:.4f})" for x in d["lubkin"][1::2])
    write("e01", r"""\begin{table}[h]\centering\footnotesize
\resizebox{\linewidth}{!}{\begin{tabular}{@{}rrr cc cc cc cc@{}}\toprule
 & & & \multicolumn{2}{c}{complex Haar} & \multicolumn{2}{c}{real Haar} & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{4th moment} \\
$M$ & $N$ & $c$ & MC & exact & MC & exact & MC & exact & real & complex \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}
\caption{E01: variance of $\langle Vx,Vy\rangle$ over """ + str(d["draws"]) + r""" draws against
\cref{prop:variances}; last columns: normalized fourth moment of the deviation (3 for a real Gaussian, 2 for $|\cdot|$ of a
complex one). Lubkin purity, Monte Carlo (exact), $d_B=16$: """ + lub + r""".}
\label{tab:e01}\end{table}
""")


def t_e02():
    d = R("e02_tms_vs_haar")
    rows = []
    for m in d["models"]:
        t, h = m["trained"], m["haar"]
        ot, oh = np.array(m["overlaps_trained"]), np.array(m["overlaps_haar"])
        rec = m["recovery"]["nonneg"]
        k = min(rec["trained"], key=lambda kk: abs(int(kk) - rec["dt_k"]))
        rows.append(f"{m['m']} & {m['p']} & {m['represented']} & {t['mean_sq_overlap']:.4f} & {h['mean_sq_overlap']:.4f} & "
                    f"{t['excess_kurtosis']:.2f} & {h['excess_kurtosis']:.2f} & {ot.max():.3f} & {oh.max():.3f} & "
                    f"{t['frame_participation']:.3f} & {k} ({rec['dt_k']:.1f}) & {rec['trained'][k]:.2f} & {rec['haar'][k]:.2f} & "
                    f"{m['tms_loss_trained']:.3f} & {m['tms_loss_haar_best_bias']:.3f} \\\\")
    write("e02", r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3.2pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}rrr cc cc cc c c cc cc@{}}\toprule
 & & & \multicolumn{2}{c}{mean $\langle w_i,w_j\rangle^2$} & \multicolumn{2}{c}{excess kurt.} & \multicolumn{2}{c}{max positive} & tight- & & \multicolumn{2}{c}{recovery at $k$} & \multicolumn{2}{c}{TMS loss} \\
$m$ & $p$ & rep. & trained & Haar & trained & Haar & trained & Haar & ness & $k$ ($k^*$) & trained & Haar & trained & Haar \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}
\caption{E02: toy models of superposition ($n=32$) against Haar frames of the same shape. ``rep.''\ = represented
features; tightness = participation ratio of the frame operator$/n$ (1 = tight). Recovery: nonnegative $\ell_1$, 30
trials at the $k$ nearest the predicted threshold $k^*$. TMS loss: trained model vs a Haar frame with the best
uniform bias. The mean overlap of every trained code equals $-1/(m-1)$ to four digits: its feature directions sum
to zero.}
\label{tab:e02}\end{table}
""")


DISPLAY = {"roberta": "RoBERTa", "glove": "GloVe"}


def t_e03():
    d = R("e03_real_embeddings")
    rows = []
    for name in ("roberta", "glove"):
        dd = d[name]; n = dd["n"]
        hf = dd["haar_frame"]
        for v in ("raw", "centred", "whitened"):
            fr_, ov = dd["versions"][v]["frame"], dd["versions"][v]["overlaps"]
            t = 4 / np.sqrt(n)
            tk = min(ov["tails"], key=lambda s: abs(float(s) - t))
            from math import erfc, sqrt
            gauss = erfc(float(tk) * sqrt(n) / sqrt(2))
            rows.append(f"{DISPLAY[name] if v == 'raw' else ''} & {v} & {ov['n_mean_sq']:.2f} & {ov['excess_kurtosis']:.2f} & "
                        f"{tk} & {ov['tails'][tk]:.1e} & {gauss:.1e} & {fr_['max_coherence']:.2f} & {hf['max_coherence']:.2f} & "
                        f"{fr_['frame_participation']:.2f} \\\\")
        rows.append(r"\addlinespace")
    rec_rows = []
    for name in ("roberta", "glove"):
        rc = d[name].get("recovery")
        if not rc:
            continue
        ks = sorted(rc["rates"]["haar"], key=int)
        for k in ks:
            rec_rows.append(f"{DISPLAY[name]} & {rc['m_sub']} & {k} ({rc['dt_k']:.0f}) & {rc['rates']['centred'][k]:.2f} & "
                            f"{rc['rates']['whitened'][k]:.2f} & {rc['rates']['haar'][k]:.2f} \\\\")
    write("e03", r"""\begin{table}[h]\centering\footnotesize
\resizebox{\linewidth}{!}{\begin{tabular}{@{}ll cc c cc cc c@{}}\toprule
 & & $n\,\E\cos^2$ & excess & & \multicolumn{2}{c}{$\Pr(|\cos|>t)$} & \multicolumn{2}{c}{max coherence} & tight- \\
model & version & (Haar $\approx1$) & kurt. & $t$ & observed & Gaussian & observed & Haar & ness \\\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule\end{tabular}}

\medskip
\resizebox{0.6\linewidth}{!}{\begin{tabular}{@{}lrr ccc@{}}\toprule
 & & & \multicolumn{3}{c}{exact $\ell_1$ recovery rate} \\
model & $m_{\rm sub}$ & $k$ ($k^*$) & centred & whitened & Haar \\\midrule
""" + "\n".join(rec_rows) + r"""
\bottomrule\end{tabular}}
\caption{E03: language-model embeddings as codes. Top: overlap statistics over $4\times10^5$ random pairs; $t$ is the tabulated threshold nearest $4/\sqrt n$ ($5.5/\sqrt n$ for RoBERTa, $5/\sqrt n$ for GloVe);
the Gaussian column is the tail of $\mathcal N(0,1/n)$; coherence over 4000 random items. Bottom: signed $\ell_1$
recovery from a random sub-dictionary of $m_{\rm sub}=2n$ items ($k^*$: predicted threshold; RoBERTa 8 trials per
entry, GloVe 30).}
\label{tab:e03}\end{table}
""")


def summarize(runs, keys, idx=-1):
    return {k: [r[idx][k] for r in runs] for k in keys}


def t_e10():
    d = R("e10_drift_conditions")
    keys = ["task", "frozen", "frozen_recal", "refit", "neurons_per_feature"]
    rows = []
    start = summarize([r["rows"] for r in d["runs"] if r["condition"] == "release"], keys, 0)
    rows.append("legible checkpoint & " + " & ".join(ms(start[k]) for k in keys) + r" \\\midrule")
    for c in ("release", "pressure", "pressure+budget", "pressure+monitor"):
        s = summarize([r["rows"] for r in d["runs"] if r["condition"] == c], keys)
        rows.append(f"{c} & " + " & ".join(ms(s[k]) for k in keys) + r" \\")
    gaps = {}
    for c in ("pressure", "pressure+monitor"):
        runs = [r["rows"] for r in d["runs"] if r["condition"] == c]
        gaps[c] = [r[-1]["task"] - r[-1]["refit"] for r in runs]
    write("e10", r"""\begin{table}[h]\centering\footnotesize
\resizebox{\linewidth}{!}{\begin{tabular}{@{}l ccccc@{}}\toprule
condition (after 6000 steps) & receiver F1 & frozen & recalibrated & refit & neurons/feature \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}
\caption{E10: four continuations of a legible code ($d=32$, $n=16$, Adam), mean $\pm$ sd over 5 seeds. ``pressure'':
feature density $p$ raised from 0.1 to 0.3; ``budget'': the $\ell_1$ legibility budget is kept; ``monitor'': the
frozen monitor's loss is added to the training loss. Receiver$-$refit gap: """ +
          f"{np.mean(gaps['pressure']):.3f}\\,$\\pm$\\,{np.std(gaps['pressure']):.3f} under pressure, "
          f"{np.mean(gaps['pressure+monitor']):.3f}\\,$\\pm$\\,{np.std(gaps['pressure+monitor']):.3f} with the monitor in the loss." + r"""}
\label{tab:e10}\end{table}
""")


def t_e11():
    d = R("e11_optimizers")
    keys = ["task", "frozen", "frozen_recal", "refit", "neurons_per_feature", "leak"]
    labels = {"adam": "Adam", "sgd": "SGD", "hybrid": "hybrid: SGD sender, Adam receiver",
              "hybrid_reverse": "reverse: Adam sender, SGD receiver", "sgd_large": "SGD, large step"}
    rows = []
    for dd in (12, 32):
        runs0 = [r["rows"] for r in d["runs"] if r["d"] == dd and r["optimizer"] == "adam"]
        s0 = summarize(runs0, keys, 0)
        rows.append(f"$d={dd}$, legible start & " + " & ".join(ms(s0[k]) for k in keys) + r" \\")
        for o in ("adam", "sgd", "hybrid", "hybrid_reverse", "sgd_large"):
            runs = [r["rows"] for r in d["runs"] if r["d"] == dd and r["optimizer"] == o]
            s = summarize(runs, keys)
            rows.append(f"\\quad {labels[o]} & " + " & ".join(ms(s[k]) for k in keys) + r" \\")
        rows.append(r"\midrule")
    write("e11", r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}l cccccc@{}}\toprule
optimizer (6000 steps, budget removed) & receiver F1 & frozen & recalibrated & refit & neurons/feature & leak \\\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule\end{tabular}}
\caption{E11: same task, legibility budget removed, five update rules; mean $\pm$ sd over 5 seeds. Learning rates:
Adam $3\times10^{-3}$, SGD $0.05$, large-step SGD $0.5$. ``leak'': fraction of sender weight energy outside each
feature's legible coordinate. $d=12<n$: empty coordinates exist; $d=32>n$: superposition.}
\label{tab:e11}\end{table}
""")


def t_e13():
    d = R("e13_saddle_hessian")
    fresh = {r["sigma"]: r for r in R("e13_fresh")}
    rows = []
    for r in sorted(d["runs"], key=lambda r: r["sigma"]):
        f = r["features"]; fr_ = fresh[r["sigma"]]
        med = lambda k: float(np.median([x[k] for x in f]))
        v = r["global_vec"]
        rows.append(f"{r['sigma']:.1f} & {fr_['gap_median']:.2f} & {np.mean(fr_['mmse_batch']):.0e} & "
                    f"{np.mean(fr_['mmse_fresh']):.0e} & {r['grad_within']:.0e} & {r['grad_leak_empty']:.0e} & "
                    f"{r['lam_global']:.0e} & {v['leak_empty']:.2f} & {med('H_ee'):.0e} & {med('schur'):.0e} & "
                    f"{np.median(fr_['immse_pred_fresh']):.0e} \\\\")
    write("e13", r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}c c cc cc cc ccc@{}}\toprule
 & sender & \multicolumn{2}{c}{residual error $\mmse$} & \multicolumn{2}{c}{gradient norm} & \multicolumn{2}{c}{global $\lambda_{\min}$} & \multicolumn{3}{c}{leak into empty coordinate (median over features)} \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-11}
$\sigma$ & gap & batch & fresh noise & within & leak, empty & value & on leak & $H_{\varepsilon\varepsilon}$ (lemma: 0) & envelope & I-MMSE \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}
\caption{E13: the legible critical point ($d=12$, $n=16$, softplus receiver), polished on a fixed batch of
""" + str(d["N"]) + r""" samples with fixed channel noise. Sender gap: $|\tanh(A_{o\ell}+a_o)-\tanh(a_o)|$, at most 2.
Residual error of the trained receiver on that batch and on fresh noise (\file{e13\_fresh.py}). Gradient norm within the
legible subspace and for leaks into empty coordinates. Global $\lambda_{\min}$ by Lanczos and the weight of its
eigenvector on leaks into empty coordinates. Last three columns: for the least saturated empty coordinate, the leak
curvature $H_{\varepsilon\varepsilon}$, the envelope (Schur-complement) curvature, and the I-MMSE prediction
$-\phi'(a_j)^2\mmse_\ell/\sigma^2$ of \cref{prop:immse} with the fresh-noise error.}
\label{tab:e13}\end{table}
""")


def t_e15():
    d = R("e15_superposition_forces")
    rows = []
    rng_ = lambda v: f"{np.min(v):.1e}--{np.max(v):.1e}"
    n, H = 16, 64
    for dd in sorted({r["d"] for r in d["runs"]}):
        rs = [r for r in d["runs"] if r["d"] == dd]
        get = lambda k: np.array([r[k] for r in rs])
        n_within = dd + n + H * n + H + dd * H + dd          # support of A, a, B, b, C, c
        rows.append(f"{dd} & {len(rs)} & {get('n_empty').mean():.1f} & {get('n_shared').mean():.1f} & "
                    f"{rng_(get('grad_within'))} & {rng_(get('grad_within') / np.sqrt(n_within))} & "
                    f"{rng_(get('grad_leak_occupied'))} & {rng_(get('grad_leak_occupied_rms'))} & "
                    f"{rng_(get('grad_leak_empty'))} \\\\")
    write("e15", r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}rr cc cc cc c@{}}\toprule
 & & \multicolumn{2}{c}{coordinates (of 16)} & \multicolumn{2}{c}{gradient within support} & \multicolumn{2}{c}{gradient, leaks into occupied} & leaks into \\\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
$d$ & seeds & empty & shared & norm & per entry & norm & per entry & empty, norm \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}
\caption{E15: is the legible code a critical point? Starting from E11's legible codes (same seeds, ReLU receiver), each
feature is restricted to its strongest coordinate (shared coordinates kept) and all parameters are polished within that
support (full batch of 6144, $\sigma=0.1$). Gradient norms, and root-mean-square per entry, in three groups of
directions; ranges are over seeds.}
\label{tab:e15}\end{table}
""")


if __name__ == "__main__":
    for f in (t_e01, t_e02, t_e03, t_e10, t_e11, t_e13, t_e15):
        try:
            f()
        except (FileNotFoundError, KeyError) as e:
            print("missing", e)
