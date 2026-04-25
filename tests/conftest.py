"""Shared pytest fixtures + common stub classes for lorebook-chunker tests.

U2 で `_StubAnalyzer` / `_ScriptedLLM` (test_ingest.py / test_smoke.py 間で重複して
いた) と `_ScriptedLLMClient` (test_wiki.py 固有) を conftest.py に集約.
pytest は `tests/` を package として扱う (`tests/__init__.py` あり / pyproject.toml
の `pythonpath = ["src"]`) ため、既存テストは `from tests.conftest import ...` で
読み込む.

また IngestConfig / tmp sample dir のボイラープレートを減らす fixture を追加:

- `ingest_config_factory` — `IngestConfig` の sane default を返す factory
- `tmp_samples_dir` — content_map を `tmp_path` 直下に書き出す factory
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pytest

from lorebook_chunker.chunker import simple_japanese_splitter
from lorebook_chunker.ingest import IngestConfig
from lorebook_chunker.llm import GenerateResult
from lorebook_chunker.schema import AnalyzerConfig, EntityMention


# ---- Stub classes ----------------------------------------------------


class _StubAnalyzer:
    """Ginza 非依存の stub. simple_japanese_splitter を使い、固定の NER を返す."""

    def __init__(self, entity_map: dict[str, list[EntityMention]] | None = None) -> None:
        self._entities = entity_map or {}

    def iter_sentences(self, text: str) -> Iterable[str]:
        return simple_japanese_splitter(text)

    def iter_entities(self, text: str) -> Iterable[EntityMention]:
        # 入力 text の最初のキー (substring) がマッチすればそのエンティティを返す
        for key, ents in self._entities.items():
            if key in text:
                yield from ents

    def tokenize_for_tfidf(self, text: str) -> list[str]:
        # シンプルな tokenization: 句点/空白で分割 + 2 文字以上
        tokens: list[str] = []
        for chunk in text.replace("\n", "。").split("。"):
            for word in chunk.split():
                w = word.strip("、.,")
                if len(w) >= 2:
                    tokens.append(w)
        # 漢字語も強引に混ぜる (文字ベースの simple tokenizer)
        for word in ["田中", "佐藤", "スカラー商事", "東京", "大阪", "プロジェクト"]:
            count = text.count(word)
            for _ in range(count):
                tokens.append(word)
        return tokens

    def save(self, path) -> None:
        # analyzer.json を書き出す (strict_match / compat_match の最小限)
        config = AnalyzerConfig(
            strict_match={
                "model_name": "stub",
                "split_mode": "STUB",
                "pos_allowlist": ["NOUN"],
                "stopwords": [],
                "lemma_rules": "stub",
                "normalization": {
                    "nfkc": True,
                    "lf_only": True,
                    "strip_trailing": True,
                    "collapse_spaces": True,
                },
                "sudachidict_binary_sha256": "stub",
                "model_checksum": "stub",
            },
            compat_match={
                "ginza": "stub-0.0",
                "spacy": "stub-0.0",
                "sudachipy": "stub-0.0",
            },
            tfidf={"min_df": 1, "max_df": 0.95},
        )
        Path(path).write_text(
            json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    def build_config(self) -> AnalyzerConfig:
        return AnalyzerConfig(
            strict_match={"model_name": "stub"},
            compat_match={"ginza": "stub-0.0"},
            tfidf={},
        )


class _ScriptedLLM:
    """IngestRunner と直結する ingest-side LLM stub.

    呼ばれるたびに `_responses` の先頭を取り出し返す (末尾に達したら OK を返す).
    wiki 生成のプロンプトは記録しない (test_wiki の `_ScriptedLLMClient` が担当).
    """

    model_id = "mock@v1"

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        self.calls += 1
        r = self._responses.pop(0) if self._responses else GenerateResult(
            text="OK",
            input_tokens=5,
            output_tokens=5,
            model_id="mock@v1",
            finish_reason="end_turn",
        )
        if isinstance(r, Exception):
            raise r
        return r


class _ScriptedLLMClient:
    """wiki 専用 LLM mock. 呼び出し順に応じて固定レスポンス or 例外を返す.

    `_ScriptedLLM` との違いは `prompts` を保持すること (wiki テストが
    プロンプト content を assert する用途).
    """

    def __init__(self, script: list, model_id: str = "mock@v1") -> None:
        self.model_id = model_id
        self._script = list(script)
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int) -> GenerateResult:
        self.call_count += 1
        self.prompts.append(prompt)
        action = self._script.pop(0) if self._script else GenerateResult(
            text="OK",
            input_tokens=10,
            output_tokens=20,
            model_id=self.model_id,
            finish_reason="end_turn",
        )
        if isinstance(action, Exception):
            raise action
        return action


# ---- Fixtures --------------------------------------------------------


@pytest.fixture
def ingest_config_factory() -> Callable[..., IngestConfig]:
    """`IngestConfig` を sane default 付きで生成する factory fixture.

    使い方:
        def test_x(tmp_path, ingest_config_factory):
            cfg = ingest_config_factory(
                tmp_path,
                input_dir=tmp_path / "in",
                output_dir=tmp_path / "out",
                recursive=True,
            )

    `tmp_path` を第一引数として渡せば `input_dir` / `output_dir` が未指定でも
    `tmp_path / "in"` と `tmp_path / "out"` に fallback する.
    それ以外の IngestConfig フィールドは overrides kwargs で上書き可.
    """

    def _factory(tmp_path: Path, **overrides: Any) -> IngestConfig:
        defaults: dict[str, Any] = {
            "input_dir": tmp_path / "in",
            "output_dir": tmp_path / "out",
            "skip_wiki": True,
            "target_chars": 30,
            "max_chunk_chars": 200,
            "min_mentions": 2,
            "min_chunks": 1,
            "show_progress": False,
        }
        defaults.update(overrides)
        return IngestConfig(**defaults)

    return _factory


@pytest.fixture
def tmp_samples_dir() -> Callable[[Path, Mapping[str, bytes | str]], Path]:
    """`content_map` の各 entry を `tmp_path / key` に書き出し、root Path を返す factory.

    - key は relative path 文字列 (例: "a.txt" や "sub/deep/b.txt").
    - value が bytes なら write_bytes、str なら utf-8 で write_text.
    - 親ディレクトリは自動生成.
    """

    def _factory(tmp_path: Path, content_map: Mapping[str, bytes | str]) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        for rel, content in content_map.items():
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
        return tmp_path

    return _factory
