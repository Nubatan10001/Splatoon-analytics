from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import japanize_matplotlib

from splatoon_analytics import SplatoonAnalytics


def safe_filename(name: str) -> str:
    name = str(name).strip().replace("\n", " ")
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def main() -> None:
    # ========= 武器設定 =========
    WEAPON_JA_TARGET = "ロングブラスター"

    # baseline(全武器平均) と weapon の最低試合数
    MIN_GAMES_BASE = 300      # 全武器の平均との差を安定させたいので少し高め推奨
    MIN_GAMES_WEAPON = 150    # 武器側のnもある程度必要
    MAX_STAGES_SHOW = None    # ステージ数が多い場合に見づらいので上位だけ表示（必要なら None）

    OUT_ROOT = Path("Analytics_output") / "delta_winrate"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ========= 1) データ読み込み =========
    sa = SplatoonAnalytics(players_dir="players-csv")
    df = sa.load_players_all(needed_cols=["mode", "stage", "weapon", "team", "win"])

    sa.load_statink_dictionaries()

    df = sa.add_is_win(df)
    df = sa.add_ja_names(df)

    # ========= 2) baseline（全武器）を mode×stage_ja で集計 =========
    base = (
        df.group_by(["mode", "stage_ja"])
          .agg(
              pl.len().alias("games_all"),
              pl.sum("is_win").alias("wins_all"),
          )
          .with_columns((pl.col("wins_all") / pl.col("games_all")).alias("win_rate_all"))
          .filter(pl.col("games_all") >= MIN_GAMES_BASE)
    )

    # ========= 3) ロングブラスターを mode×stage_ja で集計 =========
    wpn = (
        df.filter(pl.col("weapon_ja") == WEAPON_JA_TARGET)
          .group_by(["mode", "stage_ja"])
          .agg(
              pl.len().alias("games_wpn"),
              pl.sum("is_win").alias("wins_wpn"),
          )
          .with_columns((pl.col("wins_wpn") / pl.col("games_wpn")).alias("win_rate_wpn"))
          .filter(pl.col("games_wpn") >= MIN_GAMES_WEAPON)
    )

    # ========= 4) baseline と weapon を結合して Δ勝率 =========
    delta = (
        wpn.join(base, on=["mode", "stage_ja"], how="inner")
           .with_columns((pl.col("win_rate_wpn") - pl.col("win_rate_all")).alias("delta"))
    )

    # ========= 5) ルールごとにグラフ保存 =========
    modes = delta.select("mode").unique().to_series().to_list()
    modes = [m for m in ["nawabari", "area", "yagura", "hoko", "asari"] if m in modes]

    for mode in modes:
        mode_name = sa.mode_label(mode)

        mdf = (
            delta.filter(pl.col("mode") == mode)
                 .sort("delta", descending=True)
        )

        if MAX_STAGES_SHOW is not None:
            mdf = mdf.head(MAX_STAGES_SHOW)

        rows = mdf.select(["stage_ja", "delta", "games_wpn", "games_all", "win_rate_wpn", "win_rate_all"]).to_dicts()
        if not rows:
            print(f"[SKIP] {mode_name}: no data after filters")
            continue

        labels = [r["stage_ja"] for r in rows][::-1]
        values = [r["delta"] * 100 for r in rows][::-1]
        annots = [
            f"Δ={r['delta']*100:+.1f}% | wpn={r['win_rate_wpn']*100:.1f}% (n={int(r['games_wpn'])}) "
            f"| all={r['win_rate_all']*100:.1f}% (n={int(r['games_all'])})"
            for r in rows
        ][::-1]

        fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(labels))))
        ax.barh(labels, values)

        ax.axvline(0, linewidth=1)
        ax.set_title(
            f"{WEAPON_JA_TARGET}：{mode_name} ステージ別 Δ勝率（平均との差）\n"
            f"(baseline n>={MIN_GAMES_BASE}, weapon n>={MIN_GAMES_WEAPON})"
        )
        ax.set_xlabel("Δ勝率（%） = 武器勝率 - 全体勝率")

        min_v = min(values) if values else -5
        max_v = max(values) if values else 5
        pad = max(1.0, (max_v - min_v) * 0.12)
        ax.set_xlim(min_v - pad, max_v + pad)

        for i, (x, a) in enumerate(zip(values, annots)):
            if x >= 0:
                ax.text(x + 0.2, i, a, va="center", fontsize=8)
            else:
                ax.text(x - 0.2, i, a, va="center", ha="right", fontsize=8)

        plt.tight_layout()

        out_path = OUT_ROOT / f"{safe_filename(WEAPON_JA_TARGET)}_{safe_filename(mode_name)}_delta.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print("[OK]", out_path)

    print("done")


if __name__ == "__main__":
    main()
