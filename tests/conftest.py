"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture
def tiny_ja_corpus() -> list[str]:
    """3 ドキュメントの小さなコーパス (Ginza 非依存で解析ロジックを単体テストしやすくするため)."""
    return [
        "田中太郎は東京都に住んでいる。田中太郎は株式会社スカラー商事で働いている。",
        "佐藤花子は大阪府に住んでいる。佐藤花子は株式会社スカラー商事を退職した。",
        "田中太郎と佐藤花子は同じプロジェクトに参加している。プロジェクトの名前はアルファである。",
    ]
