"""chunks.jsonl のロード (query / lint 間で共有する).

F-045: 旧来 query.py と lint.py に同一ロジックが複写されていた.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from chunking.schema import ChunkFileCorruptError


def load_chunks(path: Path) -> Iterator[dict[str, Any]]:
    """chunks.jsonl を 1 行ずつ JSON としてパースして yield.

    F-028: ``json.JSONDecodeError`` は握りつぶさず ``ChunkFileCorruptError`` に変換し、
    呼び出し側 (query / lint) が exit code 3 にマップできるようにする.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ChunkFileCorruptError(
                    f"corrupt line {lineno} in {path}: {e}"
                ) from e
