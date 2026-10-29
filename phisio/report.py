"""Session report for the physiotherapist: angle curves, ROM per joint, posture time, reps, falls.

usage: uv run python -m phisio.report sessions/sessao_XXXX.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import JOINTS

LEVELS = ["OK", "ATENCAO", "RUIM"]
LEVEL_COLORS = ["#3caa46", "#ffaa00", "#dc2828"]


def main(path):
    path = Path(path)
    df = pd.read_csv(path)
    out_dir = path.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    df = df[df["person"] == 1]
    if df.empty:
        raise SystemExit("no frames with a person in this session")
    t = df["t"] - df["t"].iloc[0]

    # 1) angle curves (filtered vs raw) for joints that actually moved
    moved = [j for j in JOINTS if df[j].notna().sum() > 10 and df[j].max() - df[j].min() > 15]
    joints = moved or [j for j in JOINTS if df[j].notna().any()][:4]
    fig, axes = plt.subplots(len(joints), 1, figsize=(11, 2.4 * len(joints)), sharex=True, squeeze=False)
    for ax, j in zip(axes[:, 0], joints):
        ax.plot(t, df[f"{j}_raw"], color="#bbbbbb", lw=0.8, label="bruto")
        ax.plot(t, df[j], color="#1f2c8f", lw=1.6, label="filtrado (1€)")
        if df[f"{j}_3d"].notna().any():
            ax.plot(t, df[f"{j}_3d"], color="#ff9600", lw=1.0, label="3D (profundidade)")
        ax.set_ylabel(f"{j} (°)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("tempo (s)")
    fig.suptitle("Ângulos articulares ao longo da sessão")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "angulos.png", dpi=130)
    plt.close(fig)

    # 2) posture timeline
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for lv in range(3):
        m = df["posture"] == lv
        ax.scatter(t[m], np.full(m.sum(), lv), s=4, color=LEVEL_COLORS[lv])
    ax2 = ax.twinx()
    for metric in ["pescoco", "tronco", "ombros", "tronco_lateral"]:
        if metric in df and df[metric].notna().any():
            ax2.plot(t, df[metric], lw=0.8, label=metric)
    ax.set_yticks(range(3), LEVELS)
    ax.set_xlabel("tempo (s)")
    ax2.set_ylabel("graus")
    if ax2.lines:
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Postura classificada")
    fig.tight_layout()
    fig.savefig(out_dir / "postura.png", dpi=130)
    plt.close(fig)

    # 3) summary
    dt = np.diff(df["t"].to_numpy(), prepend=df["t"].iloc[0])
    share = {LEVELS[lv]: dt[(df["posture"] == lv).to_numpy()].sum() / max(dt.sum(), 1e-9) * 100 for lv in range(3)}
    lines = [f"# Relatório da sessão — {path.stem}", "",
             f"- Duração com pessoa detectada: {t.iloc[-1]:.0f} s ({len(df)} quadros, {df['fps'].mean():.1f} FPS médio)",
             "- Tempo em cada postura: " + ", ".join(f"{k} {v:.0f}%" for k, v in share.items()),
             f"- Repetições ({df['exercise'].iloc[-1]}): {int(df['reps'].max())}",
             f"- Eventos de queda: {int((df['fall_state'].eq('QUEDA') & ~df['fall_state'].shift().eq('QUEDA')).sum())}",
             "", "## Amplitude de movimento (ADM), ângulo filtrado", "",
             "| Articulação | mín (°) | máx (°) | ADM (°) | ADM 3D (°) |", "|---|---|---|---|---|"]
    for j in JOINTS:
        s = df[j].dropna()
        if s.empty:
            continue
        lo, hi = s.quantile(0.02), s.quantile(0.98)   # robust to single-frame outliers
        s3 = df[f"{j}_3d"].dropna()
        rom3 = f"{s3.quantile(0.98) - s3.quantile(0.02):.0f}" if len(s3) > 10 else "--"
        lines.append(f"| {j} | {lo:.0f} | {hi:.0f} | {hi - lo:.0f} | {rom3} |")
    (out_dir / "resumo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] figures and summary in {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1])
