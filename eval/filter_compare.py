"""Compare smoothing filters on an elbow-angle signal: none, moving average, EMA, Kalman and One-Euro.

Same idea as the official 1-euro interactive demo (gery.casiez.net/1euro/InteractiveDemo), but run
on our own data. Without a log it uses a synthetic signal (still -> fast flexion -> still + noise),
where the true angle is known, so we can measure the error of each filter.

usage:
  uv run python -m eval.filter_compare                                   # synthetic
  uv run python -m eval.filter_compare sessions/sessao_X.csv --joint cotovelo_D --still 2 6 --move 8 14
  add --gif docs/filtros.gif to save the animation
"""
import argparse
import math
from collections import deque

import matplotlib
import numpy as np
import pandas as pd

from phisio.one_euro_filter import OneEuroFilter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


class MovingAverage:
    def __init__(self, n=10):
        self.buf = deque(maxlen=n)

    def __call__(self, x, t):
        self.buf.append(x)
        return sum(self.buf) / len(self.buf)


class EMA:
    def __init__(self, alpha=0.15):
        self.alpha, self.y = alpha, None

    def __call__(self, x, t):
        self.y = x if self.y is None else self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class Kalman:
    """1D constant-velocity Kalman filter (state = angle, angular speed)."""

    def __init__(self, q=2000.0, r=4.0):
        self.q, self.r = q, r
        self.x = self.P = self.t = None

    def __call__(self, z, t):
        if self.x is None:
            self.x, self.P, self.t = np.array([z, 0.0]), np.eye(2) * 10.0, t
            return z
        dt = max(t - self.t, 1e-3)
        self.t = t
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        k = self.P[:, 0] / (self.P[0, 0] + self.r)          # H = [1, 0]
        self.x = self.x + k * (z - self.x[0])
        self.P = self.P - np.outer(k, self.P[0])
        return float(self.x[0])


class OneEuro:
    def __init__(self, mincutoff=1.0, beta=0.01):
        self.f = OneEuroFilter(30, mincutoff, beta, 1.0)

    def __call__(self, x, t):
        return self.f(x, t)


FILTERS = {
    "sem filtro": (lambda: (lambda x, t: x), "#9e9e9e"),
    "media movel (10)": (MovingAverage, "#e6a817"),
    "exponencial (a=0.15)": (EMA, "#8e5bd6"),
    "Kalman (vel. const.)": (Kalman, "#2f80ed"),
    "1 euro": (OneEuro, "#e0457b"),
}


def synthetic(fs=30.0, noise=3.0, seed=0):
    """Elbow held at 10 deg, fast flexion to 120 deg and back, then held; keypoint-like noise."""
    t = np.arange(0, 12, 1 / fs)
    true = np.full_like(t, 10.0)
    for t0, t1, a0, a1 in ((3, 4, 10, 120), (6, 7, 120, 10)):
        m = (t >= t0) & (t < t1)
        s = (t[m] - t0) / (t1 - t0)
        true[m] = a0 + (a1 - a0) * (1 - np.cos(np.pi * s)) / 2
    true[(t >= 4) & (t < 6)] = 120.0
    rng = np.random.default_rng(seed)
    return t, true + rng.normal(0, noise, t.size), true


def run_filters(t, x):
    out = {}
    for name, (make, _) in FILTERS.items():
        f = make()
        y, last = [], math.nan
        for ti, xi in zip(t, x):
            if math.isfinite(xi):
                last = f(float(xi), float(ti))
            y.append(last)
        out[name] = np.array(y)
    return out


def lag_ms(ref, y, fs, max_lag=20):
    """Delay (ms) that best aligns y with ref over the moving part (min. squared error after shifting)."""
    ok = np.isfinite(ref) & np.isfinite(y)
    ref, y = ref[ok], y[ok]
    best = min(range(0, max_lag + 1), key=lambda k: np.mean((ref[: len(ref) - k] - y[k:]) ** 2))
    return 1000.0 * best / fs


def metrics(t, x, ys, true=None, still=None, move=None):
    fs = 1.0 / np.median(np.diff(t))
    still_m = (t >= still[0]) & (t < still[1]) if still else None
    move_m = (t >= move[0]) & (t < move[1]) if move else None
    ref = true if true is not None else x
    rows = []
    for name, y in ys.items():
        r = {"filtro": name}
        if still_m is not None:
            r["tremor parado (desvio, graus)"] = np.nanstd(y[still_m])
        if move_m is not None:
            r["atraso no movimento (ms)"] = lag_ms(ref[move_m], y[move_m], fs)
        if true is not None:
            r["erro RMS vs real (graus)"] = float(np.sqrt(np.nanmean((y - true) ** 2)))
        rows.append(r)
    return pd.DataFrame(rows)


def animate(t, x, ys, path, true=None, fps=15, step=2, window_s=4.0):
    names = list(ys)
    fig = plt.figure(figsize=(10, 5.2), dpi=80)
    gs = fig.add_gridspec(2, len(names), height_ratios=[1.1, 1])
    arm_axes = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
    ts_ax = fig.add_subplot(gs[1, :])
    needles = []
    for ax, name in zip(arm_axes, names):
        color = FILTERS[name][1]
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.35, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=9, color=color)
        ax.plot([-1, 0], [0, 0], color="#555", lw=5, solid_capstyle="round")   # upper arm
        ax.plot(0, 0, "o", color="#333", ms=6)                                 # elbow
        ghost, = ax.plot([], [], color="#ccc", lw=3)                           # raw measurement
        line, = ax.plot([], [], color=color, lw=5, solid_capstyle="round")     # filtered forearm
        txt = ax.text(0, -0.28, "", ha="center", fontsize=9)
        needles.append((ghost, line, txt))
    if true is not None:
        ts_ax.plot(t, true, color="k", lw=1, ls="--", label="real")
    ts_ax.plot(t, x, color="#bbb", lw=1, label="medido")
    curves = {n: ts_ax.plot([], [], color=FILTERS[n][1], lw=1.6, label=n)[0] for n in names if n != "sem filtro"}
    ts_ax.set_ylabel("flexao do cotovelo (graus)")
    ts_ax.set_xlabel("tempo (s)")
    ts_ax.legend(fontsize=7, ncol=3, loc="upper left")
    lo, hi = np.nanmin(x), np.nanmax(x)
    ts_ax.set_ylim(lo - 10, hi + 25)
    fig.tight_layout()

    def tip(deg):
        # flexion 0 = straight arm pointing right; flexion bends the forearm up
        a = math.radians(deg)
        return [0, math.cos(a)], [0, math.sin(a)]

    def frame(i):
        for (ghost, line, txt), n in zip(needles, names):
            if math.isfinite(x[i]):
                ghost.set_data(*tip(x[i]))
            if math.isfinite(ys[n][i]):
                line.set_data(*tip(ys[n][i]))
                txt.set_text(f"{ys[n][i]:.0f} graus")
        for n, c in curves.items():
            c.set_data(t[: i + 1], ys[n][: i + 1])
        ts_ax.set_xlim(max(t[0], t[i] - window_s), max(t[0] + window_s, t[i]))
        return []

    anim = FuncAnimation(fig, frame, frames=range(0, len(t), step), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="session CSV from phisio.app (uses the <joint>_raw column)")
    ap.add_argument("--joint", default="cotovelo_D")
    ap.add_argument("--still", nargs=2, type=float, help="time window (s) where the arm is still")
    ap.add_argument("--move", nargs=2, type=float, help="time window (s) with fast movement")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=math.inf)
    ap.add_argument("--gif", help="save an animation (e.g. docs/filtros.gif)")
    ap.add_argument("--plot", help="save a static comparison plot (png)")
    args = ap.parse_args()

    if args.log:
        df = pd.read_csv(args.log)
        df = df[(df.t >= args.start) & (df.t <= args.end)]
        t, x, true = df.t.to_numpy(float), df[f"{args.joint}_raw"].to_numpy(float), None
    else:
        t, x, true = synthetic()
        args.still = args.still or (0.5, 2.8)
        args.move = args.move or (2.8, 7.5)

    ys = run_filters(t, x)
    print(metrics(t, x, ys, true, args.still, args.move).round(1).to_string(index=False))

    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
        if true is not None:
            ax.plot(t, true, "k--", lw=1, label="real")
        for n, y in ys.items():
            ax.plot(t, y, color=FILTERS[n][1], lw=1 if n == "sem filtro" else 1.8, label=n)
        ax.set_xlabel("tempo (s)")
        ax.set_ylabel("flexao (graus)")
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"saved {args.plot}")
    if args.gif:
        animate(t, x, ys, args.gif, true)


if __name__ == "__main__":
    main()
