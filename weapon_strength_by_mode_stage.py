import polars as pl
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

# ========= 設定 =========
INPUT_CSV = Path("players-csv/2022-09-26_players.csv")

MIN_GAMES = 10   # 少なすぎる試合数は信用しにくいので除外
TOP_K = 5        # 各 stage の TOP何件を出すか（要件：TOP5）

MODE_JA = {
    "nawabari": "ナワバリ",
    "area": "ガチエリア",
    "yagura": "ガチヤグラ",
    "hoko": "ガチホコ",
    "asari": "ガチアサリ",
}

# ========= 1) 読み込み =========
df = pl.read_csv(str(INPUT_CSV))

# ========= 2) 勝ちフラグ =========
df = df.with_columns(
    (pl.col("team") == pl.col("win")).cast(pl.Int32).alias("is_win")
)

# ========= 3) ルール×ステージ×武器で集計 =========
agg = (
    df.group_by(["mode", "stage", "weapon"])
      .agg(
          pl.count().alias("games"),
          pl.sum("is_win").alias("wins"),
      )
      .with_columns(
          (pl.col("wins") / pl.col("games")).alias("win_rate")
      )
)

# ========= 4) 各 mode×stage で TOP5 を作る（ただし n>=MIN_GAMES） =========
top = (
    agg.filter(pl.col("games") >= MIN_GAMES)
       .sort(["mode", "stage", "win_rate", "games"], descending=[False, False, True, True])
       .group_by(["mode", "stage"])
       .head(TOP_K)
)

# ========= 5) 「ステージごと」に横棒グラフを表示（保存しない） =========
modes = sorted(top.select("mode").unique().to_series().to_list())

for mode in modes:
    mode_df = top.filter(pl.col("mode") == mode)
    mode_name = MODE_JA.get(mode, mode)

    stages = sorted(mode_df.select("stage").unique().to_series().to_list())

    for stage in stages:
        stage_df = (
            mode_df.filter(pl.col("stage") == stage)
                   .sort(["win_rate", "games"], descending=[True, True])
        )

        # python側へ取り出し
        rows = stage_df.select(["weapon", "win_rate", "games", "wins"]).to_dicts()

        # 横棒（上に強い武器が来るようにする）
        weapons = [r["weapon"] for r in rows][::-1]
        rates   = [r["win_rate"] * 100 for r in rows][::-1]
        annots  = [f"n={r['games']} (w={r['wins']})" for r in rows][::-1]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(weapons))))
        ax.barh(weapons, rates)

        ax.set_title(f"{mode_name} / {stage}：勝率TOP{TOP_K}武器（n>={MIN_GAMES}）")
        ax.set_xlabel("勝率（%）")
        ax.set_xlim(0, 100)

        # 値ラベル
        for i, (x, a) in enumerate(zip(rates, annots)):
            ax.text(x + 0.5, i, a, va="center", fontsize=9)

        plt.tight_layout()
        plt.show()      # ← 画像保存せずターミナル(実行環境)で表示
        plt.close(fig)
