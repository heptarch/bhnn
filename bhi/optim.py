"""Optimizers with per-parameter rules, so that e.g. the sender can be trained with
plain SGD while the receiver uses Adam (the "hybrid" optimizer of Part II)."""
import numpy as np


class PerParam:
    """rules: dict name -> ("adam", lr) or ("sgd", lr).  Updates params in place."""

    def __init__(self, params, rules, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.rules, self.b1, self.b2, self.eps = params, rules, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            kind, lr = self.rules[k]
            if kind == "sgd":
                self.p[k] -= lr * g
            elif kind == "adam":
                self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
                mh = self.m[k] / (1 - self.b1 ** self.t)
                vh = self.v[k] / (1 - self.b2 ** self.t)
                self.p[k] -= lr * mh / (np.sqrt(vh) + self.eps)
            else:
                raise ValueError(kind)


SENDER = ("A", "a")
RECEIVER = ("B", "b", "C", "c")


def adam(params, lr=3e-3):
    return PerParam(params, {k: ("adam", lr) for k in params})


def sgd(params, lr=0.05):
    return PerParam(params, {k: ("sgd", lr) for k in params})


def hybrid(params, sgd_lr=0.05, adam_lr=3e-3):
    """Plain SGD on the channel (sender) parameters, Adam on everything else."""
    rules = {k: (("sgd", sgd_lr) if k in SENDER else ("adam", adam_lr)) for k in params}
    return PerParam(params, rules)
