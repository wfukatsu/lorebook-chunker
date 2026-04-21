"""`ingest` CLI の本体: 全 Unit を結線し、staging dir で atomic swap する."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from chunking.chunker import Chunker
from chunking.llm import LLMClient, LLMPermanentError, get_client
from chunking.ner import (
    DEFAULT_MIN_CHUNKS,
    DEFAULT_MIN_MENTIONS,
    DEFAULT_TARGET_LABELS,
    aggregate_entities,
    attach_entities_to_chunks,
    extract_entities_per_chunk,
    sanitize_entity_filename,
)
from chunking.normalize import normalize_text
from chunking.schema import ChunkRecord
from chunking.tfidf import TfidfBuilder
from chunking.wiki import (
    WikiGenerator,
    WikiGeneratorConfig,
    WikiStats,
    compute_analyzer_json_hash,
)

logger = logging.getLogger(__name__)


# ---- Analyzer protocol + factory -----------------------------------------


class IngestAnalyzer(Protocol):
    """IngestRunner が必要とする analyzer の最小インターフェース.

    JapaneseAnalyzer が実装するが、テストでは stub に差し替え可能.
    """

    def iter_sentences(self, text: str) -> Iterable[str]: ...

    def iter_entities(self, text: str) -> Iterable[Any]: ...

    def tokenize_for_tfidf(self, text: str) -> list[str]: ...

    def save(self, path: str | os.PathLike[str]) -> None: ...


AnalyzerFactory = Callable[[Path | None], IngestAnalyzer]
LLMClientFactory = Callable[[str, dict[str, Any] | None], LLMClient]


def default_analyzer_factory(existing_path: Path | None) -> IngestAnalyzer:
    """既存 analyzer.json があれば load、無ければ新規生成."""
    from chunking.analyzer import JapaneseAnalyzer

    if existing_path is not None and existing_path.exists():
        return JapaneseAnalyzer.load_and_verify(existing_path)
    return JapaneseAnalyzer()


# ---- Runner config --------------------------------------------------------


@dataclass
class IngestConfig:
    input_dir: Path
    output_dir: Path
    skip_wiki: bool = False
    max_llm_calls: int | None = None
    force_regenerate: bool = False
    retry_failed: bool = False
    llm_backend: str = "anthropic"
    target_chars: int = 500
    overlap_chars: int = 100
    max_chunk_chars: int = 1500
    top_keywords: int = 10
    min_mentions: int = DEFAULT_MIN_MENTIONS
    min_chunks: int = DEFAULT_MIN_CHUNKS
    target_labels: tuple[str, ...] = DEFAULT_TARGET_LABELS


@dataclass
class IngestResult:
    exit_code: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    chunks_generated: int = 0
    total_input_files: int = 0
    skipped_files: int = 0
    wiki_stats: WikiStats | None = None


# ---- Runner -----------------------------------------------------------


class IngestRunner:
    def __init__(
        self,
        cfg: IngestConfig,
        *,
        analyzer_factory: AnalyzerFactory = default_analyzer_factory,
        llm_factory: LLMClientFactory = get_client,
    ) -> None:
        self.cfg = cfg
        self._analyzer_factory = analyzer_factory
        self._llm_factory = llm_factory

    def run(self) -> IngestResult:
        result = IngestResult(exit_code=0)
        input_files = _collect_input_files(self.cfg.input_dir)
        result.total_input_files = len(input_files)
        if not input_files:
            result.exit_code = 2
            result.errors.append(
                f"no .txt files under {self.cfg.input_dir}"
            )
            return result

        # 早期 fail-fast: LLM バックエンド生成 (OpenAI なら即例外)
        if not self.cfg.skip_wiki:
            try:
                llm = self._llm_factory(self.cfg.llm_backend, {})
            except LLMPermanentError as e:
                result.exit_code = 3
                result.errors.append(f"LLM backend unavailable: {e}")
                return result
        else:
            llm = _NoopLLM()

        # staging dir
        staging = self.cfg.output_dir.with_name(self.cfg.output_dir.name + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)

        try:
            # 1. 既存 output_dir から manifest を読む
            existing_manifest = self.cfg.output_dir / "entities" / "manifest.json"
            # staging 側の entities/manifest.json に pre-populate する (wiki generator が load するので)
            if existing_manifest.exists():
                (staging / "entities").mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing_manifest, staging / "entities" / "manifest.json")
                # 既存 entity .md もキャッシュコピー (wiki generator が書き直さない場合に残る)
                src_ents = self.cfg.output_dir / "entities"
                if src_ents.exists():
                    for md in src_ents.glob("*.md"):
                        shutil.copy2(md, staging / "entities" / md.name)

            # 2. analyzer をロード or 新規生成
            existing_analyzer = self.cfg.output_dir / "analyzer.json"
            try:
                analyzer = self._analyzer_factory(existing_analyzer if existing_analyzer.exists() else None)
            except Exception as e:
                result.exit_code = 4
                result.errors.append(f"analyzer init failed: {e}")
                return result

            # 3. 入力正規化 + チャンク化
            chunker = Chunker(
                splitter=analyzer.iter_sentences,
                target_chars=self.cfg.target_chars,
                overlap_chars=self.cfg.overlap_chars,
                max_chunk_chars=self.cfg.max_chunk_chars,
            )
            all_chunks: list[ChunkRecord] = []
            for file_path in input_files:
                try:
                    raw = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    result.skipped_files += 1
                    result.warnings.append(f"utf-8 decode failed, skipped: {file_path}")
                    continue
                if not raw.strip():
                    result.skipped_files += 1
                    result.warnings.append(f"empty file, skipped: {file_path}")
                    continue
                normalized = normalize_text(raw)
                relative = str(file_path.relative_to(self.cfg.input_dir))
                for chunk in chunker.chunk_document(
                    relative, normalized, input_dir=str(self.cfg.input_dir)
                ):
                    all_chunks.append(chunk)
            for w in chunker.warnings:
                result.warnings.append(f"chunker/{w.kind}: {w.detail}")

            if not all_chunks:
                result.exit_code = 5
                result.errors.append("chunking produced 0 chunks")
                return result

            # 4. TF-IDF
            tfidf = TfidfBuilder(
                analyzer=analyzer.tokenize_for_tfidf,
                top_keywords=self.cfg.top_keywords,
            )
            texts = [c.text for c in all_chunks]
            matrix, vocab, idf = tfidf.fit_transform(texts)
            keywords = tfidf.top_keywords_per_chunk(matrix, vocab)
            for chunk, kws in zip(all_chunks, keywords):
                chunk.top_keywords = kws

            # 5. NER + aggregate
            entities_per_chunk = extract_entities_per_chunk(
                analyzer, all_chunks, self.cfg.target_labels
            )
            attach_entities_to_chunks(all_chunks, entities_per_chunk)
            aggregates, agg_stats = aggregate_entities(
                all_chunks,
                entities_per_chunk,
                min_mentions=self.cfg.min_mentions,
                min_chunks=self.cfg.min_chunks,
            )

            # 6. 書き出し: analyzer.json / vocab.npz / chunks.jsonl
            analyzer.save(staging / "analyzer.json")
            tfidf.save(staging / "vocab.npz", matrix, vocab, idf)
            with (staging / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for chunk in all_chunks:
                    f.write(json.dumps(chunk.to_jsonable(), ensure_ascii=False))
                    f.write("\n")

            # 7. エンティティ wiki
            wiki_stats = WikiStats()
            if not self.cfg.skip_wiki:
                wiki_cfg = WikiGeneratorConfig(
                    entities_dir=staging / "entities",
                    manifest_path=staging / "entities" / "manifest.json",
                    max_llm_calls=self.cfg.max_llm_calls,
                    skip_wiki=False,
                    force_regenerate=self.cfg.force_regenerate,
                    retry_failed=self.cfg.retry_failed,
                )
                analyzer_json_hash = compute_analyzer_json_hash(staging / "analyzer.json")
                wiki = WikiGenerator(
                    llm_client=llm,
                    config=wiki_cfg,
                    analyzer_json_hash=analyzer_json_hash,
                    ginza_model_version=_ginza_model_version(analyzer),
                    chunks=all_chunks,
                )
                try:
                    wiki_stats = wiki.generate_all(aggregates)
                except LLMPermanentError as e:
                    result.exit_code = 6
                    result.errors.append(f"wiki generation aborted: {e}")
                    return result

            # 8. index.md
            _write_index_md(staging / "index.md", staging / "entities" / "manifest.json")

            # 9. log.md
            _append_log_md(
                staging / "log.md",
                input_files=input_files,
                skipped_files=result.skipped_files,
                chunks_generated=len(all_chunks),
                agg_stats=agg_stats,
                wiki_stats=wiki_stats,
                warnings=result.warnings,
                ginza_model=_ginza_model_version(analyzer),
                llm_model=getattr(llm, "model_id", "noop"),
            )

            # 10. atomic swap
            _atomic_swap(self.cfg.output_dir, staging)
            result.chunks_generated = len(all_chunks)
            result.wiki_stats = wiki_stats
            return result
        except Exception as e:
            logger.exception("ingest failed")
            result.exit_code = 10
            result.errors.append(f"ingest failure: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            return result


# ---- helpers --------------------------------------------------------


def _collect_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    return sorted(input_dir.glob("*.txt"))


def _ginza_model_version(analyzer: Any) -> str:
    try:
        cfg = analyzer.build_config()
        name = cfg.strict_match.get("model_name", "unknown")
        ver = cfg.compat_match.get("ginza", "unknown")
        return f"{name}@{ver}"
    except Exception:
        return "unknown"


def _atomic_swap(target: Path, staging: Path) -> None:
    """target を staging の内容で置き換える. 途中キャンセル耐性は best-effort.

    手順:
      1. target が存在するなら target.with_suffix('.backup') に move
      2. staging を target に rename
      3. backup を削除
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(target.name + ".backup")
        if backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
    try:
        staging.rename(target)
    except OSError:
        # rename が跨ぎで失敗する場合は copytree + rmtree
        shutil.copytree(staging, target)
        shutil.rmtree(staging)
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _write_index_md(path: Path, manifest_path: Path) -> None:
    if not manifest_path.exists():
        path.write_text(
            "# Entity Index\n\n(エンティティ wiki は生成されていません)\n",
            encoding="utf-8",
        )
        return
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = list((data.get("entries") or {}).values())
    # NER ラベル別に分類 + 五十音順ソート
    by_label: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("status") != "success":
            continue
        by_label.setdefault(e["ner_label"], []).append(e)
    for label in by_label:
        by_label[label].sort(key=lambda x: x["entity_name"])

    lines: list[str] = ["# Entity Index", ""]
    for label in sorted(by_label.keys()):
        lines.append(f"## {label}")
        lines.append("")
        for e in by_label[label]:
            fname = sanitize_entity_filename(e["ner_label"], e["entity_name"])
            lines.append(
                f"- [{e['entity_name']}](entities/{fname}) "
                f"(mentions: {e['mention_count']}, chunks: {e['chunk_count']})"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_log_md(
    path: Path,
    *,
    input_files: list[Path],
    skipped_files: int,
    chunks_generated: int,
    agg_stats,
    wiki_stats: WikiStats,
    warnings: list[str],
    ginza_model: str,
    llm_model: str,
) -> None:
    header = "# Ingest log\n\n" if not path.exists() else ""
    entry = [
        f"## Run {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- input files: {len(input_files)} (skipped: {skipped_files})",
        f"- chunks generated: {chunks_generated}",
        f"- entities detected: {agg_stats.total_detected_entities}",
        f"- entities accepted (>= thresholds): {agg_stats.accepted_entities}",
        f"- entities skipped by threshold: {agg_stats.skipped_entities}",
        f"- wiki attempted: {wiki_stats.attempted}",
        f"- wiki succeeded: {wiki_stats.succeeded}",
        f"- wiki failed: {wiki_stats.failed}",
        f"- wiki budget_skipped: {wiki_stats.budget_skipped}",
        f"- wiki cached (skipped): {wiki_stats.skipped_cached}",
        f"- LLM input tokens total: {wiki_stats.total_input_tokens}",
        f"- LLM output tokens total: {wiki_stats.total_output_tokens}",
        f"- ginza model: {ginza_model}",
        f"- llm model: {llm_model}",
    ]
    if warnings:
        entry.append("- warnings:")
        for w in warnings:
            entry.append(f"  - {w}")
    entry.append("")
    with path.open("a", encoding="utf-8") as f:
        if header:
            f.write(header)
        f.write("\n".join(entry))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


# ---- NoopLLM for --skip-wiki path -----------------------------------


class _NoopLLM:
    """--skip-wiki 時に LLM を呼ばない dummy client."""

    model_id = "noop@skip-wiki"

    def generate(self, prompt: str, max_tokens: int):  # pragma: no cover
        raise RuntimeError("NoopLLM.generate should not be called when skip_wiki=True")


# ---- CLI entry ------------------------------------------------------


def run_ingest(args: argparse.Namespace) -> int:
    cfg = IngestConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        skip_wiki=getattr(args, "skip_wiki", False),
        max_llm_calls=getattr(args, "max_llm_calls", None),
        force_regenerate=getattr(args, "force_regenerate", False),
        retry_failed=getattr(args, "retry_failed", False),
        llm_backend=getattr(args, "llm_backend", "anthropic"),
    )
    # `--force-regenerate` と `--retry-failed` の共存: force 優先、retry は警告
    if cfg.force_regenerate and cfg.retry_failed:
        logger.warning(
            "--retry-failed and --force-regenerate both set; "
            "--force-regenerate takes precedence and --retry-failed is a no-op"
        )
        cfg.retry_failed = False

    # default_analyzer_factory is captured at module-lookup time to allow monkeypatching
    result = IngestRunner(
        cfg,
        analyzer_factory=default_analyzer_factory,
        llm_factory=get_client,
    ).run()
    for w in result.warnings:
        print(f"[warn] {w}", file=sys.stderr)
    for e in result.errors:
        print(f"[error] {e}", file=sys.stderr)
    if result.exit_code == 0:
        print(
            f"ingest OK: {result.chunks_generated} chunks from "
            f"{result.total_input_files - result.skipped_files} files -> {cfg.output_dir}",
            file=sys.stderr,
        )
    return result.exit_code
