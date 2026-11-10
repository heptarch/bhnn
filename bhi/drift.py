"""Shared machinery for the Part II legibility-drift experiments."""
import numpy as np
from . import channel as ch
from . import optim
from .probes import (fit_probe, probe_f1, hoyer, active_coords, participation_ratio,
                     fit_recalibration, column_hoyer, neurons_per_feature)

DEFAULT = dict(d=32, n=16, H=64, sigma=0.10, p=0.10, batch=256,
               legible_lam=1.0, legible_steps=4000, legible_lr=3e-3)


def train(P, opt, steps, cfg, rng, p=None, lam=0.0, monitor=None, mu=0.0,
          checkpoints=(), on_checkpoint=None):
    p = cfg["p"] if p is None else p
    for t in range(1, steps + 1):
        x = ch.sample_features(cfg["batch"], cfg["d"], p, rng)
        xi = rng.standard_normal((cfg["batch"], cfg["n"]))
        _, _, g = ch.loss_and_grads(P, x, xi, cfg["sigma"], lam=lam, monitor=monitor, mu=mu)
        opt.step(g)
        if t in checkpoints and on_checkpoint is not None:
            on_checkpoint(t)
    return P


def make_legible(cfg, rng):
    """Train with an l1 budget in the public frame, then fit the frozen monitor."""
    P = ch.init_params(cfg["d"], cfg["n"], cfg["H"], rng)
    opt = optim.adam(P, cfg["legible_lr"])
    train(P, opt, cfg["legible_steps"], cfg, rng, lam=cfg["legible_lam"])
    x = ch.sample_features(8000, cfg["d"], cfg["p"], rng)
    h = ch.forward(P, x, np.zeros((8000, cfg["n"])), 0.0)["h"]
    monitor = fit_probe(h, x, steps=1500)
    owner = np.argmax(np.abs(P["A"]), axis=0)      # public coordinate owned by each feature
    return P, monitor, owner


def leak_fraction(P, owner):
    """Fraction of sender weight-energy outside each feature's legible coordinate."""
    A2 = P["A"] ** 2
    own = A2[owner, np.arange(A2.shape[1])]
    return float(1 - own.sum() / A2.sum())


def evaluate(P, monitor, owner, cfg, rng, p=None, n_eval=4000, refit_steps=1500):
    p = cfg["p"] if p is None else p
    d, n, s = cfg["d"], cfg["n"], cfg["sigma"]
    x = ch.sample_features(n_eval, d, p, rng)
    xi = rng.standard_normal((n_eval, n))
    f = ch.forward(P, x, xi, s)
    h = f["h"]
    task = ch.f1_score((f["logits"] > 0).astype(float), x)
    xt = ch.sample_features(n_eval, d, p, rng)
    ht = ch.forward(P, xt, np.zeros((n_eval, n)), 0.0)["h"]
    refit = fit_probe(ht, xt, steps=refit_steps)
    recal = fit_recalibration(monitor, ht, xt)
    h0 = np.tanh(P["a"])                            # message for the all-zero feature vector
    return dict(task=task, hoyer=hoyer(h), hoyer_centered=hoyer(h - h0), k=active_coords(h),
                pr=participation_ratio(h),
                frozen=probe_f1(monitor, h, x), frozen_recal=probe_f1(recal, h, x),
                refit=probe_f1(refit, h, x),
                col_hoyer=column_hoyer(P["A"]), neurons_per_feature=neurons_per_feature(P["A"]),
                leak=leak_fraction(P, owner))


def run_condition(P0, monitor, owner, cfg, rng, make_opt, steps, p=None, lam=0.0, mu=0.0,
                  checkpoints=(0, 250, 500, 1000, 2000, 4000, 6000)):
    P = {k: v.copy() for k, v in P0.items()}
    opt = make_opt(P)
    rows = []

    def record(t):
        r = evaluate(P, monitor, owner, cfg, rng, p=p)
        r["step"] = t
        rows.append(r)

    if 0 in checkpoints:
        record(0)
    train(P, opt, steps, cfg, rng, p=p, lam=lam, monitor=monitor, mu=mu,
          checkpoints=set(checkpoints), on_checkpoint=record)
    return rows, P
