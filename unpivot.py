import re
from pathlib import Path
import polars as pl

INPUT_DIR = Path("battle-results-csv")
OUT_DIR = Path("players-csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLAYER_COL_RE = re.compile(r"^(A[1-4]|B[1-4])-(.+)$")

def unpivot_one_file(input_path: Path, output_path: Path) -> None:
    # 1) 読み込み
    df = pl.read_csv(str(input_path))

    # 2) プレイヤー列/試合共通列を分離
    player_cols = [c for c in df.columns if PLAYER_COL_RE.match(c)]
    match_cols  = [c for c in df.columns if c not in player_cols]

    # 3) 試合ID付与
    df = df.with_row_index("match_id")

    # 4) スロット一覧
    slots = sorted({PLAYER_COL_RE.match(c).group(1) for c in player_cols})

    # 5) アンピボット（1行=1試合 → 1行=1プレイヤー）
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

    # 7) CSV出力
    df_long.write_csv(str(output_path))


def main():
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        print(f"入力CSVが見つかりません: {INPUT_DIR}")
        return

    print(f"対象ファイル数: {len(files)}")

    done = 0
    skipped = 0
    failed = 0

    for input_path in files:
        date_stem = input_path.stem
        output_path = OUT_DIR / f"{date_stem}_players.csv"

        # 既に存在するならスキップ（不要ならこのifごと消してOK）
        if output_path.exists():
            print(f"[SKIP] {input_path.name} -> {output_path.name} (already exists)")
            skipped += 1
            continue

        try:
            unpivot_one_file(input_path, output_path)
            print(f"[OK]   {input_path.name} -> {output_path.name}")
            done += 1
        except Exception as e:
            print(f"[FAIL] {input_path.name} -> {e}")
            failed += 1

    print("\n==== 結果 ====")
    print(f"完了: {done}")
    print(f"スキップ: {skipped}")
    print(f"失敗: {failed}")


if __name__ == "__main__":
    main()
