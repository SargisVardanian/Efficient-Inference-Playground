from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_barplot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out: str | Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

