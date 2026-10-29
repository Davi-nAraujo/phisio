"""Posture classification accuracy from a labeled session.

While recording, press 7/8/9 in the app to mark the posture you are *actually* doing
(OK / ATENCAO / RUIM) and 0 to clear. This script compares those labels with the classifier.

usage: uv run python eval/posture_eval.py sessions/sessao_XXXX.csv
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVELS = ["OK", "ATENCAO", "RUIM"]


def main(path):
    df = pd.read_csv(path)
    df = df[(df["person"] == 1) & df["gt"].notna()]
    if df.empty:
        raise SystemExit("no labeled frames (press 7/8/9 during the session)")
    y, p = df["gt"].astype(int).to_numpy(), df["posture"].astype(int).to_numpy()
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    acc = (y == p).mean()
    print(f"frames: {len(y)}  acurácia: {acc * 100:.1f}%\n")
    print("matriz de confusão (linhas = real, colunas = previsto)")
    print(pd.DataFrame(cm, index=LEVELS, columns=LEVELS))
    for i, name in enumerate(LEVELS):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else float("nan")
        rec = tp / cm[i].sum() if cm[i].sum() else float("nan")
        print(f"{name:8s} precisão {prec:.2f}  revocação {rec:.2f}")

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks(range(3), LEVELS)
    ax.set_yticks(range(3), LEVELS)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(f"Postura — acurácia {acc * 100:.0f}%")
    fig.tight_layout()
    # two-camera log: also score each camera alone, to show what the fusion adds
    for col in ("posture_A", "posture_B"):
        if col in df:
            print(f"acurácia só com a câmera {col[-1]}: {(df[col].astype(int).to_numpy() == y).mean() * 100:.1f}%")
    out = path.replace(".csv", "_confusao.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
