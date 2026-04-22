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
from typing import Any, Callable, Iterable, NoReturn, Protocol

from lorebook_chunker.chunker import Chunker
from lorebook_chunker.cli import IDENTITY_BANNER
from lorebook_chunker.llm import LLMClient, LLMPermanentError, get_client
from lorebook_chunker.ner import (
    AggregationStats,
    DEFAULT_MIN_CHUNKS,
    DEFAULT_MIN_MENTIONS,
    DEFAULT_TARGET_LABELS,
    aggregate_entities,
    attach_entities_to_chunks,
    extract_entities_per_chunk,
    sanitize_entity_filename,
)
from lorebook_chunker.normalize import normalize_text
from lorebook_chunker.progress import ProgressReporter
from lorebook_chunker.schema import ChunkRecord
from lorebook_chunker.tfidf import TfidfBuilder
from lorebook_chunker.wiki import (
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


ANALYZER_BACKEND_MODEL = {
    "electra": "ja_ginza_electra",
    "ginza": "ja_ginza",
}


def default_analyzer_factory(
    existing_path: Path | None,
    *,
    analyzer_backend: str = "electra",
    device: str = "cpu",
) -> IngestAnalyzer:
    """既存 analyzer.json があれば load、無ければ backend/device 指定で新規生成.

    load_and_verify は保存された model_name を尊重するため、既存 analyzer.json が
    ある場合は CLI 引数の analyzer_backend は **無視される** (一貫性優先).
    """
    from lorebook_chunker.analyzer import JapaneseAnalyzer

    if existing_path is not None and existing_path.exists():
        return JapaneseAnalyzer.load_and_verify(existing_path)
    model_name = ANALYZER_BACKEND_MODEL.get(analyzer_backend, "ja_ginza_electra")
    return JapaneseAnalyzer(model_name=model_name, device=device)


def _supports_doc_api(analyzer: Any) -> bool:
    """analyzer が Doc-level fast path (nlp_pipe / tokens_from_doc / entities_from_doc /
    batch_iter_sentences) を実装しているかを判定.
    本番 JapaneseAnalyzer では True, test stub では False → 旧パス fallback."""
    return all(
        hasattr(analyzer, name)
        for name in (
            "nlp_pipe",
            "tokens_from_doc",
            "entities_from_doc",
            "batch_iter_sentences",
        )
    )


def _supports_single_pass(analyzer: Any) -> bool:
    """analyzer が `analyze_documents` (単一パス ELECTRA) を実装しているか.

    本番 JapaneseAnalyzer が実装する. 非対応の stub は 2 パス経路に fallback する.
    """
    return hasattr(analyzer, "analyze_documents")


def _resolve_pipe_config(*, device: str = "cpu") -> tuple[int, int]:
    """nlp.pipe の (batch_size, n_process) を解決.

    既定値は bench_pipe2.py の結果 (光学コーパス 825 chunk / 11 コア MacBook):
    - batch_size=32 以上は CPU 飽和で効果頭打ち
    - n_process 4〜6 で ~1.7〜1.9x、8 以降は IPC コストで頭打ち
    env で上書きしたいときは LOREBOOK_CHUNKER_BATCH_SIZE / _N_PROCESS.

    device が cpu 以外 (mps/cuda) の場合は n_process を強制的に 1 に固定する.
    PyTorch の GPU コンテキストはプロセス間で共有できないため、spawn された
    ワーカーはデバイスロストで落ちるか CPU に fallback するため意味がない.
    """
    import os

    try:
        batch_size = max(1, int(os.environ.get("LOREBOOK_CHUNKER_BATCH_SIZE", "32")))
    except ValueError:
        batch_size = 32
    if device != "cpu":
        return batch_size, 1
    try:
        n_process = max(1, int(os.environ.get("LOREBOOK_CHUNKER_N_PROCESS", "4")))
    except ValueError:
        n_process = 4
    return batch_size, n_process


def _slice_single_pass(
    analyses: list[Any],
    chunks: list[ChunkRecord],
    chunk_analysis_idx: list[int],
    target_labels: frozenset[str] | set[str],
    *,
    progress: Any = None,
) -> tuple[list[list[str]], list[list[Any]]]:
    """`analyze_documents` の結果をチャンク範囲でスライスし、
    (tokens_per_chunk, entities_per_chunk) を返す.

    analyze_documents は ``DocumentAnalysis`` を返すので、各 chunk の
    [char_start, char_end) を bisect でクエリすれば再推論不要. ELECTRA は
    1 度しか走らない.
    """
    tokens_out: list[list[str]] = []
    entities_out: list[list[Any]] = []
    if progress is not None:
        progress.start("チャンク切り出し (1-pass)", total=len(chunks))
    labels_set: frozenset[str] | set[str] = (
        target_labels if isinstance(target_labels, (frozenset, set))
        else frozenset(target_labels)
    )
    try:
        for chunk, ai in zip(chunks, chunk_analysis_idx):
            analysis = analyses[ai]
            tokens_out.append(analysis.tokens_in_range(chunk.char_start, chunk.char_end))
            entities_out.append(
                analysis.entities_in_range(chunk.char_start, chunk.char_end, labels_set)
            )
            if progress is not None:
                progress.tick()
    finally:
        if progress is not None:
            progress.end()
    return tokens_out, entities_out


def _compute_tokens_and_entities_via_pipe(
    analyzer: Any,
    texts: list[str],
    target_labels: set[str] | frozenset[str],
    *,
    batch_size: int = 32,
    n_process: int = 1,
    progress: Any = None,
) -> tuple[list[list[str]], list[list[Any]]]:
    """chunk text 列を一括で nlp.pipe に流し、Doc から token 列 + entity 列を同時抽出する.

    - nlp.pipe のメリット: spaCy が ELECTRA transformer を batch forward する.
    - Doc 再利用のメリット: 旧実装では tokenize_for_tfidf と iter_entities が同じ
      chunk text に対して独立に _nlp(text) を呼んでおり、2 倍の推論コストが掛かっていた.
    - n_process > 1 で spaCy native multiprocessing. order は保存されるので
      入力順=出力順で ChunkRecord と 1:1 対応可能.
    - target_labels フィルタは従来の extract_entities_per_chunk と同等挙動.
    - progress が与えられれば本関数が start/tick/end を管理する.
    """
    tokens_out: list[list[str]] = []
    entities_out: list[list[Any]] = []
    if progress is not None:
        progress.start("chunk TF-IDF + NER 抽出", total=len(texts))
    try:
        for doc in analyzer.nlp_pipe(texts, batch_size=batch_size, n_process=n_process):
            tokens_out.append(analyzer.tokens_from_doc(doc))
            entities_out.append(
                [m for m in analyzer.entities_from_doc(doc) if m.ner_label in target_labels]
            )
            if progress is not None:
                progress.tick()
    finally:
        if progress is not None:
            progress.end()
    return tokens_out, entities_out


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
    llm_model: str | None = None
    # wiki 生成の同時呼出数. None ならバックエンド別既定値 (anthropic=5, ollama=3) を使う.
    llm_parallelism: int | None = None
    target_chars: int = 500
    overlap_chars: int = 100
    max_chunk_chars: int = 1500
    top_keywords: int = 10
    min_mentions: int = DEFAULT_MIN_MENTIONS
    min_chunks: int = DEFAULT_MIN_CHUNKS
    target_labels: tuple[str, ...] = DEFAULT_TARGET_LABELS
    # 日本語 NLP backend/device (Tier 2.2 MPS / Tier 3.1 軽量モデル)
    analyzer_backend: str = "electra"
    device: str = "cpu"
    # 進捗表示 (stderr). --quiet で False, CLI で明示的に --no-progress が付けば False.
    show_progress: bool = True


def _resolve_wiki_parallelism(
    backend: str, explicit: int | None
) -> int:
    """llm_parallelism の未指定時バックエンド別既定値.

    - anthropic: 公式 concurrency limit 内で安全な 5 (rate limit 余裕あり).
    - ollama: OLLAMA_NUM_PARALLEL の典型値 3 (M2 Pro 32GB で qwen3:8b が安全).
    - その他: 1 (conservative).
    """
    if explicit is not None and explicit > 0:
        return int(explicit)
    if backend == "anthropic":
        return 5
    if backend == "ollama":
        return 3
    return 1


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
        analyzer_factory: "Callable[[Path | None], IngestAnalyzer]" = default_analyzer_factory,
        llm_factory: "Callable[[str, dict[str, Any] | None], LLMClient]" = get_client,
    ) -> None:
        self.cfg = cfg
        self._analyzer_factory = analyzer_factory
        self._llm_factory = llm_factory

    def run(self) -> IngestResult:
        result = IngestResult(exit_code=0)
        progress = ProgressReporter(enabled=self.cfg.show_progress)
        input_files = _collect_input_files(self.cfg.input_dir)
        result.total_input_files = len(input_files)
        if not input_files:
            result.exit_code = 2
            result.errors.append(
                f"no .txt files under {self.cfg.input_dir}"
            )
            return result
        progress.info(
            f"{len(input_files)} files / backend={self.cfg.analyzer_backend} "
            f"device={self.cfg.device}"
        )

        # 早期 fail-fast: LLM バックエンド生成 (OpenAI なら即例外)
        if not self.cfg.skip_wiki:
            llm_config: dict[str, Any] = {}
            if self.cfg.llm_model:
                llm_config["model"] = self.cfg.llm_model
            try:
                llm = self._llm_factory(self.cfg.llm_backend, llm_config)
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
                # 既存 entity .md もキャッシュコピー (wiki generator が書き直さない場合に残る).
                # hardlink は試みない: wiki generator が Path.write_text で上書きする場合、
                # hardlink 経由で元 output_dir の md を truncate してしまい atomic-swap の
                # 途中失敗耐性が崩れる (backup からの復元が不可能になる).
                src_ents = self.cfg.output_dir / "entities"
                if src_ents.exists():
                    for md in src_ents.glob("*.md"):
                        shutil.copy2(md, staging / "entities" / md.name)

            # 2. analyzer をロード or 新規生成
            existing_analyzer = self.cfg.output_dir / "analyzer.json"
            progress.info("analyzer 初期化中 (spaCy / モデルロード)...")
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
            pipe_batch_size, pipe_n_process = _resolve_pipe_config(device=self.cfg.device)
            # 3a. ファイル読み込み + normalize (CPU 軽い) を先に一括で済ませる.
            file_records: list[tuple[Path, str, str]] = []  # (path, relative, normalized)
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
                file_records.append((file_path, relative, normalized))

            # 3b. 単一パス fast path:
            #     `analyze_documents` が提供されるなら、全ファイルを 1 回の nlp.pipe に
            #     束ねて「文境界 + TF-IDF lemma + 絶対 offset 付き entity」をまとめて回収する.
            #     旧実装は per-file ループで sentences を取った後、さらに chunk 群に対して
            #     もう一度 ELECTRA を通していた (実測で NLP 時間の ~50% が重複).
            single_pass_analyses: list[Any] | None = None
            if _supports_single_pass(analyzer) and file_records:
                texts_for_analysis = [r[2] for r in file_records]
                single_pass_analyses = analyzer.analyze_documents(
                    texts_for_analysis,
                    batch_size=pipe_batch_size,
                    n_process=pipe_n_process,
                    progress=progress,
                    progress_label="文書解析 (nlp.pipe: sents+lemma+entities)",
                )
                precomputed_sents: list[list[str] | None] = [a.sentences for a in single_pass_analyses]
            elif _supports_doc_api(analyzer) and file_records:
                # 互換 path: Doc API は持つが単一パス API はない analyzer
                texts_for_split = [r[2] for r in file_records]
                precomputed_sents = analyzer.batch_iter_sentences(
                    texts_for_split,
                    batch_size=pipe_batch_size,
                    n_process=pipe_n_process,
                    progress=progress,
                    progress_label="文境界抽出 (nlp.pipe)",
                )
            else:
                precomputed_sents = [None] * len(file_records)

            # 3c. chunker.chunk_document を実行 (fast path では iter_sentences を呼び出さない).
            #     chunks と「元ファイルの解析結果」の対応を保つため、chunks -> analysis idx
            #     のマッピングも同時に作る (単一パス経路でチャンクごとに tokens/entities を切り出すため).
            progress.info(f"chunker 実行中 ({len(file_records)} files)...")
            all_chunks: list[ChunkRecord] = []
            chunk_analysis_idx: list[int] = []
            for file_idx, ((file_path, relative, normalized), sents) in enumerate(
                zip(file_records, precomputed_sents)
            ):
                for chunk in chunker.chunk_document(
                    relative,
                    normalized,
                    input_dir=str(self.cfg.input_dir),
                    sentences=sents,
                ):
                    all_chunks.append(chunk)
                    chunk_analysis_idx.append(file_idx)
            for w in chunker.warnings:
                result.warnings.append(f"chunker/{w.kind}: {w.detail}")

            # F-001: row_index はコーパス全体に対してグローバルに振る.
            # chunker 側では -1 sentinel が入っている.
            for i, chunk in enumerate(all_chunks):
                chunk.row_index = i

            if not all_chunks:
                result.exit_code = 5
                result.errors.append("chunking produced 0 chunks")
                return result

            # 4-5. TF-IDF + NER
            # 単一パス path: ファイル単位 DocumentAnalysis から chunk 範囲でスライス.
            # 旧 Doc API path: 全 chunk text を改めて nlp.pipe に流す (ELECTRA 2 回目).
            # stub path: iter_sentences / iter_entities / tokenize_for_tfidf を個別呼び出し.
            tfidf = TfidfBuilder(
                analyzer=analyzer.tokenize_for_tfidf,
                top_keywords=self.cfg.top_keywords,
            )
            texts = [c.text for c in all_chunks]
            target_labels_set = frozenset(self.cfg.target_labels)
            if single_pass_analyses is not None:
                tokens_per_chunk, entities_per_chunk = _slice_single_pass(
                    single_pass_analyses,
                    all_chunks,
                    chunk_analysis_idx,
                    target_labels_set,
                    progress=progress,
                )
                progress.info("TF-IDF 行列を構築中...")
                matrix, vocab, idf = tfidf.fit_transform_pretokenized(tokens_per_chunk)
            elif _supports_doc_api(analyzer):
                tokens_per_chunk, entities_per_chunk = _compute_tokens_and_entities_via_pipe(
                    analyzer,
                    texts,
                    target_labels_set,
                    batch_size=pipe_batch_size,
                    n_process=pipe_n_process,
                    progress=progress,
                )
                progress.info("TF-IDF 行列を構築中...")
                matrix, vocab, idf = tfidf.fit_transform_pretokenized(tokens_per_chunk)
            else:
                progress.info("TF-IDF 行列を構築中 (旧パス, stub 用)...")
                matrix, vocab, idf = tfidf.fit_transform(texts)
                entities_per_chunk = extract_entities_per_chunk(
                    analyzer, all_chunks, self.cfg.target_labels
                )
            keywords = tfidf.top_keywords_per_chunk(matrix, vocab)
            for chunk, kws in zip(all_chunks, keywords):
                chunk.top_keywords = kws

            attach_entities_to_chunks(all_chunks, entities_per_chunk)
            progress.info("エンティティ集計中...")
            aggregates, agg_stats = aggregate_entities(
                all_chunks,
                entities_per_chunk,
                min_mentions=self.cfg.min_mentions,
                min_chunks=self.cfg.min_chunks,
            )

            # 6. 書き出し: analyzer.json / vocab.npz / chunks.jsonl
            progress.info(
                f"artifact 書き出し中 (chunks={len(all_chunks)}, vocab={len(vocab)}, "
                f"entities={len(aggregates)})"
            )
            analyzer.save(staging / "analyzer.json")
            tfidf.save(staging / "vocab.npz", matrix, vocab, idf)
            with (staging / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for chunk in all_chunks:
                    f.write(json.dumps(chunk.to_jsonable(), ensure_ascii=False))
                    f.write("\n")

            # 7. エンティティ wiki
            wiki_stats = WikiStats()
            if not self.cfg.skip_wiki:
                wiki_parallelism = _resolve_wiki_parallelism(
                    self.cfg.llm_backend, self.cfg.llm_parallelism
                )
                progress.info(
                    f"wiki 生成開始 (対象 {len(aggregates)} エンティティ, "
                    f"LLM 呼出発生, parallelism={wiki_parallelism})"
                )
                wiki_cfg = WikiGeneratorConfig(
                    entities_dir=staging / "entities",
                    manifest_path=staging / "entities" / "manifest.json",
                    max_llm_calls=self.cfg.max_llm_calls,
                    skip_wiki=False,
                    force_regenerate=self.cfg.force_regenerate,
                    retry_failed=self.cfg.retry_failed,
                    parallelism=wiki_parallelism,
                    progress=progress,
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
                    # F-015: systemic-abort 時は staging の manifest を sibling dir に退避.
                    _preserve_failed_manifest(self.cfg.output_dir, staging)
                    return result

                # 8. index.md (--skip-wiki 時には作らない — F-009)
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
            return result
        finally:
            # F-005/F-058: 早期 return (exit_code 5/6/10 等) 時にも staging をクリーンアップ.
            # atomic swap が成功した場合は staging は既に rename 済みで存在しない.
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


# ---- helpers --------------------------------------------------------


def _collect_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    return sorted(input_dir.glob("*.txt"))


def _ginza_model_version(analyzer: IngestAnalyzer) -> str:
    """analyzer.build_config() から model_name@ginza_version を取り出す.

    F-021: 失敗時は "unknown" を返すが、黙って返さず warning を出すことで
    source_hash の無音破壊 (Ginza 更新が検知されない) を目視可能にする.
    """
    try:
        cfg = analyzer.build_config()  # type: ignore[attr-defined]
        name = cfg.strict_match.get("model_name", "unknown")
        ver = cfg.compat_match.get("ginza", "unknown")
        return f"{name}@{ver}"
    except Exception as e:
        logger.warning("ginza model version lookup failed: %s", e)
        return "unknown"


def _preserve_failed_manifest(output_dir: Path, staging: Path) -> None:
    """systemic-abort 時に staging の manifest を sibling dir へ退避 (F-015).

    staging 自体は caller の finally で削除されるが、退避先は残す.
    """
    manifest = staging / "entities" / "manifest.json"
    if not manifest.exists():
        return
    sibling = output_dir.with_name(output_dir.name + ".failed") / "entities"
    try:
        sibling.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, sibling / "manifest.json")
    except OSError as e:  # pragma: no cover
        logger.warning("failed to preserve manifest for inspection: %s", e)


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
    agg_stats: AggregationStats,
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

    def generate(self, prompt: str, max_tokens: int) -> NoReturn:  # pragma: no cover
        raise RuntimeError("NoopLLM.generate should not be called when skip_wiki=True")


# ---- CLI entry ------------------------------------------------------


INGEST_EXIT_CODES = """\
exit codes:
  0  success
  2  no input .txt files under input_dir
  3  LLM backend unavailable (permanent error at init)
  4  analyzer init failed
  5  chunking produced zero chunks
  6  wiki generation aborted (systemic failure detected)
  10 unexpected error (see log / stderr)
"""


def run_ingest(args: argparse.Namespace) -> int:
    quiet = getattr(args, "quiet", False)
    if not quiet:
        print(IDENTITY_BANNER, file=sys.stderr)

    cfg = IngestConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        skip_wiki=getattr(args, "skip_wiki", False),
        max_llm_calls=getattr(args, "max_llm_calls", None),
        force_regenerate=getattr(args, "force_regenerate", False),
        retry_failed=getattr(args, "retry_failed", False),
        llm_backend=getattr(args, "llm_backend", "anthropic"),
        llm_model=getattr(args, "llm_model", None),
        llm_parallelism=getattr(args, "llm_parallelism", None),
        analyzer_backend=getattr(args, "analyzer_backend", "electra"),
        device=getattr(args, "device", "cpu"),
        # 進捗: --quiet で強制 off, --no-progress でも off, それ以外は on.
        show_progress=not (
            getattr(args, "quiet", False) or getattr(args, "no_progress", False)
        ),
    )
    # `--force-regenerate` と `--retry-failed` の共存: force 優先、retry は警告
    if cfg.force_regenerate and cfg.retry_failed:
        logger.warning(
            "--retry-failed and --force-regenerate both set; "
            "--force-regenerate takes precedence and --retry-failed is a no-op"
        )
        cfg.retry_failed = False

    # analyzer_backend / device を closure で factory に束ねる.
    # IngestRunner は `factory(path)` としか呼ばないため、test stub の lambda 署名
    # (`lambda path: ...`) と互換性を保つ必要がある. 既存 analyzer.json がある場合は
    # load_and_verify が保存 model_name を尊重するので backend 引数は無視される.
    _backend = cfg.analyzer_backend
    _device = cfg.device

    def _factory(existing_path: Path | None) -> IngestAnalyzer:
        # monkeypatch で差し替えられた lambda 等は kwargs を受け取れないため
        # まず kwargs 付きで呼び、TypeError なら位置引数のみで再試行する.
        factory = default_analyzer_factory  # 解決時に monkeypatch を尊重
        try:
            return factory(
                existing_path,
                analyzer_backend=_backend,
                device=_device,
            )
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                return factory(existing_path)
            raise

    result = IngestRunner(
        cfg,
        analyzer_factory=_factory,
        llm_factory=get_client,
    ).run()
    for w in result.warnings:
        print(f"[warn] {w}", file=sys.stderr)
    for e in result.errors:
        print(f"[error] {e}", file=sys.stderr)

    # F-036: success time に ingest_result.json を書き出し. JSON format 時は stdout にも出力.
    summary = {
        "exit_code": result.exit_code,
        "chunks_generated": result.chunks_generated,
        "total_input_files": result.total_input_files,
        "skipped_files": result.skipped_files,
        "warnings": list(result.warnings),
        "errors": list(result.errors),
        "output_dir": str(cfg.output_dir),
    }
    fmt = getattr(args, "format", "human")
    if result.exit_code == 0:
        try:
            (cfg.output_dir / "ingest_result.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:  # pragma: no cover
            logger.warning("failed to write ingest_result.json: %s", e)
        if fmt == "json":
            print(json.dumps(summary, ensure_ascii=False))
        elif not quiet:
            print(
                f"ingest OK: {result.chunks_generated} chunks from "
                f"{result.total_input_files - result.skipped_files} files -> {cfg.output_dir}",
                file=sys.stderr,
            )
    else:
        if fmt == "json":
            print(json.dumps(summary, ensure_ascii=False))
    return result.exit_code
