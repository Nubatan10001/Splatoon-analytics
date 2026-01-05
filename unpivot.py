import re
import polars as pl

INPUT_PATH = "battle-results-csv/2022-09-26.csv"
OUTPUT_PATH = "2022-09-26_players.csv"

PLAYER_COL_RE = re.compile(r"^(A[1-4]|B[1-4])-(.+)$")

# 1) 読み込み（元CSVは変更されない）
df = pl.read_csv(INPUT_PATH)

# 2) プレイヤー列/試合共通列を分離
player_cols = [c for c in df.columns if PLAYER_COL_RE.match(c)]
match_cols  = [c for c in df.columns if c not in player_cols]

# 3) 試合ID付与（後で確認・集計しやすい）
df = df.with_row_index("match_id")

# 4) スロット一覧（存在するものだけ）
slots = sorted({PLAYER_COL_RE.match(c).group(1) for c in player_cols})

# 5) アンピボット（1行=試合 → 1行=プレイヤー）
pieces = []
for slot in slots:
    slot_cols = [c for c in player_cols if c.startswith(f"{slot}-")]
    rename_map = {c: PLAYER_COL_RE.match(c).group(2) for c in slot_cols}

    one = (
        df.select(["match_id", *match_cols, *slot_cols])
          .rename(rename_map)
          .with_columns(pl.lit(slot).alias("player"))
          .with_columns([
              pl.when(pl.col("player").str.starts_with("A"))
                .then(pl.lit("alpha"))
                .otherwise(pl.lit("bravo"))
                .alias("team"),
              pl.col("player").str.slice(1).cast(pl.Int32).alias("player_no"),
          ])
    )
    pieces.append(one)

df_long = pl.concat(pieces, how="vertical").sort(["match_id", "team", "player_no"])

# 6) medal列はA1のみ値を残し、それ以外はNullにする
medal_cols = [
    "medal1-grade", "medal1-name",
    "medal2-grade", "medal2-name",
    "medal3-grade", "medal3-name",
]
existing_medal_cols = [c for c in medal_cols if c in df_long.columns]

if existing_medal_cols:
    df_long = df_long.with_columns([
        pl.when(pl.col("player") == "A1")
          .then(pl.col(c))
          .otherwise(None)
          .alias(c)
        for c in existing_medal_cols
    ])

# 7) 簡単な確認（先頭だけ表示）
print(df_long.head(16))

# 8) CSV出力（別名で安全）
df_long.write_csv(OUTPUT_PATH)
print(f"出力完了: {OUTPUT_PATH}")
