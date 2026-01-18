# 共通クラス（読み込み/辞書/集計/保存）

from __future__ import annotations

import json
import glob
from pathlib import Path
from urllib.request import urlopen

import polars as pl


MODE_JA = {
    "nawabari": "ナワバリ",
    "area": "ガチエリア",
    "yagura": "ガチヤグラ",
    "hoko": "ガチホコ",
    "asari": "ガチアサリ",
}


class SplatoonAnalytics:
    def __init__(
        self,
        players_dir: str | Path = "players-csv",
        infer_schema_length: int = 200,
    ) -> None:
        self.players_dir = Path(players_dir)
        self.infer_schema_length = infer_schema_length

        self.stage_ja: dict[str, str] = {}
        self.weapon_ja: dict[str, str] = {}

    # -------- data loading --------
    def load_players_all(self, needed_cols=None) -> pl.DataFrame:
        """
        players-csv/*_players.csv を全部読み込み（高速: scan_csv）
        """
        if needed_cols is None:
            needed_cols = ["mode", "stage", "weapon", "team", "win"]

        files = sorted(glob.glob(str(self.players_dir / "*_players.csv")))
        if not files:
            raise FileNotFoundError(f"No *_players.csv found under: {self.players_dir}")

        df = (
            pl.scan_csv(
                files,
                infer_schema_length=self.infer_schema_length,
                ignore_errors=True,
            )
            .select(needed_cols)
            .collect(streaming=True)
        )
        return df

    # -------- dictionaries --------
    def load_statink_dictionaries(self) -> None:
        """
        stat.ink APIからステージ名/武器名の日本語辞書を取得
        """
        STAGE_API = "https://stat.ink/api/v3/stage"
        WEAPON_API = "https://stat.ink/api/v3/weapon"

        def fetch_json(url: str):
            with urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))

        stages = fetch_json(STAGE_API)
        weapons = fetch_json(WEAPON_API)

        self.stage_ja = {s["key"]: s["name"].get("ja_JP", s["key"]) for s in stages}
        self.weapon_ja = {w["key"]: w["name"].get("ja_JP", w["key"]) for w in weapons}

    # -------- helpers --------
    @staticmethod
    def add_is_win(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col("team") == pl.col("win")).cast(pl.Int32).alias("is_win"))

    def add_ja_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Polars古め互換: replace + coalesce で辞書に無い場合は元値を残す
        """
        if not self.stage_ja or not self.weapon_ja:
            raise RuntimeError("Dictionaries not loaded. Call load_statink_dictionaries() first.")

        return df.with_columns([
            pl.coalesce([pl.col("stage").replace(self.stage_ja), pl.col("stage")]).alias("stage_ja"),
            pl.coalesce([pl.col("weapon").replace(self.weapon_ja), pl.col("weapon")]).alias("weapon_ja"),
        ])

    def mode_label(self, mode: str) -> str:
        return MODE_JA.get(mode, mode)
