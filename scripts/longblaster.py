from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import japanize_matplotlib

from scripts.splatoon_analytics import SplatoonAnalytics


def safe_filename(name: str) -> str:
    name = str(name).strip().replace("\n", " ")
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def main():
    WEAPON_JA_TARGET = "ロングブラスター"
    MIN_GAMES_STAGE = 1

    OUT_ROOT = Path("Analytics_output") / "weapon_stage_winrate"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sa = SplatoonAnalytics(players_dir="players-csv")
    df = sa.load_players_all(needed_cols=["mode", "stage", "weapon", "team", "win"])
    sa.load_statink_dictionaries()

    df = sa.add_is_win(df)
    df = sa.add_ja_names(df)

    lb = df.filter(pl.col("weapon_ja") == WEAPON_JA_TARGET)

    # mode × stageで勝率
    agg = (
        lb.group_by(["mode", "stage_ja"])
          .agg(
              pl.len().alias("games"),
              pl.sum("is_win").alias("wins"),
          )
          .with_columns((pl.col("wins") / pl.col("games")).alias("win_rate"))
          .filter(pl.col("games") >= MIN_GAMES_STAGE)
    )

    modes = agg.select("mode").unique().to_series().to_list()
    modes = [m for m in ["nawabari", "area", "yagura", "hoko", "asari"] if m in modes]

    for mode in modes:
        mode_name = sa.mode_label(mode)
        mdf = agg.filter(pl.col("mode") == mode).sort("win_rate", descending=True)

        rows = mdf.select(["stage_ja", "win_rate", "games", "wins"]).to_dicts()
        if not rows:
            print(f"[SKIP] {mode_name}: no stage with n>={MIN_GAMES_STAGE}")
            continue

        labels = [r["stage_ja"] for r in rows][::-1]
        values = [r["win_rate"] * 100 for r in rows][::-1]
        annots = [f"n={int(r['games'])} (w={int(r['wins'])})" for r in rows][::-1]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(labels))))
        ax.barh(labels, values)
        ax.set_title(f"{WEAPON_JA_TARGET}：{mode_name} ステージ別勝率（n>={MIN_GAMES_STAGE}）")
        ax.set_xlabel("勝率（%）")
        ax.set_xlim(0, 100)

        for i, (x, a) in enumerate(zip(values, annots)):
            ax.text(x + 0.5, i, a, va="center", fontsize=9)

        plt.tight_layout()

        out_path = OUT_ROOT / f"{safe_filename(WEAPON_JA_TARGET)}_{safe_filename(mode_name)}_n{MIN_GAMES_STAGE}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print("[OK]", out_path)

    print("done")


if __name__ == "__main__":
    main()
