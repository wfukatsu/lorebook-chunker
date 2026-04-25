"""U8: 大容量 corpus を決定論的に生成するヘルパ.

`test_large_corpus.py` から呼ばれ、`RUN_LARGE=1` 環境変数で gated される.
stub analyzer / 実 GiNZA のどちらでも動くよう、`_StubAnalyzer.tokenize_for_tfidf`
が拾うハードコードトークン (`田中`, `佐藤`, `スカラー商事`, `東京`, `大阪`,
`プロジェクト`) を十分な密度で含める.

生成は `random.Random(seed)` で決定論 — 同じ seed なら同一バイト列を出力する
(determinism テストの前提).
"""
from __future__ import annotations

import random
from pathlib import Path

# stub analyzer の tokenize_for_tfidf が拾うトークンを主成分にする.
# これらが chunks.jsonl の top_keywords に現れ、entity aggregation の母集団に
# なる. 実 GiNZA 側でも PROPN として認識される日本語固有名詞.
_SUBJECTS = ["田中", "佐藤", "スカラー商事", "東京支社", "大阪支社"]
_OBJECTS = [
    "プロジェクト",
    "合併交渉",
    "新規事業",
    "顧客対応",
    "監査対応",
    "海外展開",
    "採用計画",
]
_VERBS = [
    "担当した",
    "推進した",
    "報告した",
    "調整した",
    "完了した",
    "延期した",
    "評価した",
]
_PLACES = ["東京", "大阪", "福岡", "名古屋", "札幌"]


def _make_sentence(rng: random.Random) -> str:
    """1 文 (句点込み) を返す. 長さは概ね 30〜60 文字程度."""
    subj = rng.choice(_SUBJECTS)
    obj = rng.choice(_OBJECTS)
    verb = rng.choice(_VERBS)
    place = rng.choice(_PLACES)
    # 決定論的かつ entity dense な日本語テンプレート.
    templates = [
        f"{subj}は{place}で{obj}を{verb}。",
        f"{subj}と{rng.choice(_SUBJECTS)}は{obj}について協議した。",
        f"{place}の{subj}は{obj}の責任者として{verb}。",
        f"{obj}に関する会議で{subj}が{verb}。",
    ]
    return rng.choice(templates)


def _make_file_content(rng: random.Random, target_bytes: int) -> str:
    """target_bytes 前後になるまで文を連結. UTF-8 で厳密に数える."""
    parts: list[str] = []
    total = 0
    # 段落を 3-5 文で構成し、段落間は `\n\n` で区切る (sudachi byte-limit split
    # の段落境界 priority にも乗る).
    while total < target_bytes:
        para_len = rng.randint(3, 5)
        para = "".join(_make_sentence(rng) for _ in range(para_len))
        parts.append(para)
        total += len(para.encode("utf-8")) + 2  # +2 for "\n\n"
    return "\n\n".join(parts) + "\n"


def generate_large_corpus(
    root: Path,
    *,
    n_files: int = 100,
    avg_size_kb: int = 5,
    seed: int = 42,
) -> list[Path]:
    """`n_files` 個の `.txt` を `root` 配下に生成し、Path の list を返す.

    - 各ファイルは概ね `avg_size_kb` KB (UTF-8 byte count).
    - `seed` が同じなら同一バイト列を再生成する (determinism 保証).
    - ファイル名は `corpus_{i:04d}.txt` (sort 順が決定論的).
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    target_bytes = avg_size_kb * 1024
    created: list[Path] = []
    for i in range(n_files):
        # per-file seed を master rng から派生させる. こうしておくと
        # 1 ファイルの生成ロジック変更が他のファイルに波及しない.
        file_rng = random.Random(rng.random())
        content = _make_file_content(file_rng, target_bytes)
        path = root / f"corpus_{i:04d}.txt"
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created


__all__ = ["generate_large_corpus"]
