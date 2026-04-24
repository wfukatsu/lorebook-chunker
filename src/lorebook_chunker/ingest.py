"""`ingest` CLI の本体: 全 Unit を結線し、staging dir で atomic swap する."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, NoReturn, Protocol

from lorebook_chunker.chunker import Chunker
from lorebook_chunker.cli import IDENTITY_BANNER
from lorebook_chunker.encoding import detect_encoding
from lorebook_chunker.errors import (
    AnalyzerInitError,
    AtomicSwapError,
    ConfigError,
    EncodingError,
    LLMBackendUnavailableError,
    LorebookError,
    WikiGenerationError,
    ZeroChunksError,
)
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
from lorebook_chunker.schema import ChunkRecord, SkipReport
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
    # U2: 入力探索. recursive=True なら rglob, それ以外は glob.
    # globs は default ("*.txt",) で後方互換. tuple なら内部 API から直接指定可能、
    # CLI から渡すときは cli.py 側でカンマ区切り文字列をタプルに parse する.
    recursive: bool = False
    globs: tuple[str, ...] = ("*.txt",)
    # U3: エンコーディング. "auto" (既定) で staged detection (utf-8-sig →
    # charset-normalizer → EncodingError). 明示指定した文字列は strict decode.
    encoding: str = "auto"


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
    # U1: errors は LorebookError.to_jsonable() 出力 (dict) または
    # unclassified Exception を表す dict。旧 format (文字列) は廃止し、
    # run_report.json への machine-readable dump を容易にする。
    errors: list[dict[str, Any]] = field(default_factory=list)
    chunks_generated: int = 0
    total_input_files: int = 0
    # U4: 構造化された skip レコード. `len(skips)` が旧 `skipped_files` int と
    # 互換. 外部 caller は `result.skipped_files` (property) を読むだけで従来
    # どおり int カウントを取得できる. mutation は `result.skips.append(...)`
    # 経由で行う.
    skips: list[SkipReport] = field(default_factory=list)
    wiki_stats: WikiStats | None = None
    # U5: 6 stable phase name (analyzer_init / chunking / tfidf / ner / wiki / swap)
    # → duration seconds の dict. IngestRunner.run の末尾で
    # `progress.phase_timings()` から写される. 失敗 phase も finally 経由で
    # 記録される (wiki で例外発生 → wiki の duration は >0 で残る).
    phase_timings: dict[str, float] = field(default_factory=dict)
    # U5: NER 集計で閾値通過したユニークエンティティ数 (AggregationStats.accepted_entities).
    # 現状は log.md にしか出ないので run_report.json 用に持ち上げる.
    entities_generated: int = 0

    @property
    def skipped_files(self) -> int:
        """U4: `len(self.skips)` の後方互換 alias.

        旧 `IngestResult` には `skipped_files: int` フィールドがあり、一部の
        call-site (tests / CLI summary / log.md) は read 専用で参照していた.
        U4 で構造化 `skips` を導入した後も read 側契約を維持するための
        derived property.
        """
        return len(self.skips)


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
        # staging は外側 try/finally で cleanup するため、try の外で宣言する.
        staging: Path | None = None
        # U5: 現在「通過中」の phase 名と開始時刻を保持する.
        # `_begin_phase(name)` で start、正常通過時は `_end_phase()` で記録、
        # 例外時は outermost except/finally で `_end_phase(force=True)` を
        # 呼び、失敗 phase の partial duration を残す.
        _active_phase: list[str | None] = [None]
        _active_phase_t0: list[float] = [0.0]

        def _begin_phase(name: str) -> None:
            _active_phase[0] = name
            _active_phase_t0[0] = time.perf_counter()

        def _end_phase() -> None:
            if _active_phase[0] is not None:
                progress._record_duration(
                    _active_phase[0], time.perf_counter() - _active_phase_t0[0]
                )
                _active_phase[0] = None

        try:
            # U3: encoding 引数の妥当性を早期検証. "auto" 以外は Python codec
            # 名として解決可能でなければ ConfigError (exit 2).
            _validate_encoding_option(self.cfg.encoding)

            input_files = _collect_input_files(
                self.cfg.input_dir,
                globs=self.cfg.globs,
                recursive=self.cfg.recursive,
            )
            result.total_input_files = len(input_files)
            if not input_files:
                # U1: "設定が入力を生まなかった" は ConfigError (exit 2).
                # 既存テストが assert する "no .txt" substring を message 冒頭に
                # 残しつつ、context に structured な値を持たせる.
                # default globs ("*.txt",) 時は従来どおり "no .txt files" 表示.
                label = (
                    ".txt"
                    if tuple(self.cfg.globs) == ("*.txt",)
                    else ", ".join(self.cfg.globs)
                )
                raise ConfigError(
                    f"no {label} files under {self.cfg.input_dir}",
                    reason="no_input_files",
                    input_dir=str(self.cfg.input_dir),
                    globs=list(self.cfg.globs),
                    recursive=self.cfg.recursive,
                )
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
                    # preflight の LLM 初期化失敗 → exit 3 (既存契約).
                    # LLMPermanentError 自体は LorebookError 派生だが exit code
                    # は 10 なので、ここで明示的に LLMBackendUnavailableError
                    # にラップする (preflight vs runtime の区別を保つ).
                    raise LLMBackendUnavailableError(
                        f"LLM backend unavailable: {e}",
                        backend=self.cfg.llm_backend,
                        reason=type(e).__name__,
                    ) from e
            else:
                llm = _NoopLLM()

            # staging dir
            staging = self.cfg.output_dir.with_name(self.cfg.output_dir.name + ".staging")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=False)

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
            # U5: analyzer_init phase boundary.
            existing_analyzer = self.cfg.output_dir / "analyzer.json"
            progress.info("analyzer 初期化中 (spaCy / モデルロード)...")
            _begin_phase("analyzer_init")
            try:
                analyzer = self._analyzer_factory(existing_analyzer if existing_analyzer.exists() else None)
            except LorebookError:
                # すでに分類済み (AnalyzerInitError 派生の
                # AnalyzerVersionMismatchError / AnalyzerNEUnavailableError 等)
                # はそのまま伝播させる.
                raise
            except Exception as e:
                # 未分類の runtime error を AnalyzerInitError (exit 4) に wrap.
                raise AnalyzerInitError(
                    f"analyzer init failed: {e}",
                    reason="init_exception",
                    cause=type(e).__name__,
                ) from e
            _end_phase()

            # 3. 入力正規化 + チャンク化
            # U5: chunking phase boundary. ZeroChunksError / 単一パス解析での
            # 例外はすべて outermost except で捕捉され、`_end_phase(force=True)`
            # 相当の finally で partial duration が記録される.
            _begin_phase("chunking")
            chunker = Chunker(
                splitter=analyzer.iter_sentences,
                target_chars=self.cfg.target_chars,
                overlap_chars=self.cfg.overlap_chars,
                max_chunk_chars=self.cfg.max_chunk_chars,
            )
            pipe_batch_size, pipe_n_process = _resolve_pipe_config(device=self.cfg.device)
            # 3a. ファイル読み込み + normalize (CPU 軽い) を先に一括で済ませる.
            # U3: `detect_encoding` で staged detection (utf-8-sig → charset-normalizer).
            # U4: skip 発生時は `result.skips` に構造化 `SkipReport` を append し、
            # 後段で `skipped_files.jsonl` として emit する. 既存テストが assertion
            # している warning 文字列 ("empty file" / "utf-8 decode failed") は
            # log.md 互換のため result.warnings にも同内容を残す.
            file_records: list[tuple[Path, str, str]] = []  # (path, relative, normalized)
            for file_path in input_files:
                try:
                    raw, _actual_encoding = detect_encoding(
                        file_path, override=self.cfg.encoding
                    )
                except EncodingError as e:
                    result.skips.append(_skip_report_from_encoding_error(file_path, e))
                    result.warnings.append(f"utf-8 decode failed, skipped: {file_path}")
                    continue
                except UnicodeDecodeError as e:
                    # detect_encoding は EncodingError に wrap する設計だが、
                    # 将来的な変更 / monkeypatch 経由の直接 raise に備えた safety net.
                    result.skips.append(
                        SkipReport(
                            path=str(file_path),
                            reason="encoding_decode_failed",
                            detail=str(e),
                            encoding_attempted=getattr(e, "encoding", None),
                            size_bytes=_safe_stat_size(file_path),
                        )
                    )
                    result.warnings.append(f"utf-8 decode failed, skipped: {file_path}")
                    continue
                except FileNotFoundError as e:
                    result.skips.append(
                        SkipReport(
                            path=str(file_path),
                            reason="file_not_found",
                            detail=str(e),
                            encoding_attempted=None,
                            size_bytes=None,
                        )
                    )
                    result.warnings.append(f"file not found, skipped: {file_path}")
                    continue
                except PermissionError as e:
                    result.skips.append(
                        SkipReport(
                            path=str(file_path),
                            reason="permission_denied",
                            detail=str(e),
                            encoding_attempted=None,
                            size_bytes=_safe_stat_size(file_path),
                        )
                    )
                    result.warnings.append(f"permission denied, skipped: {file_path}")
                    continue
                if not raw.strip():
                    result.skips.append(
                        SkipReport(
                            path=str(file_path),
                            reason="empty_file",
                            detail=None,
                            encoding_attempted=None,
                            size_bytes=_safe_stat_size(file_path),
                        )
                    )
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
                raise ZeroChunksError(
                    "chunking produced 0 chunks",
                    reason="zero_chunks",
                    total_input_files=result.total_input_files,
                    skipped_files=result.skipped_files,
                )
            _end_phase()  # U5: chunking phase 終了

            # 4-5. TF-IDF + NER
            # 単一パス path: ファイル単位 DocumentAnalysis から chunk 範囲でスライス.
            # 旧 Doc API path: 全 chunk text を改めて nlp.pipe に流す (ELECTRA 2 回目).
            # stub path: iter_sentences / iter_entities / tokenize_for_tfidf を個別呼び出し.
            # U5: tfidf phase boundary. inner progress.start/end calls in
            # `_slice_single_pass` / `_compute_tokens_and_entities_via_pipe`
            # は `_durations` には触らず、outer phase の記録と独立に動く
            # (それぞれ別キーで上書きされる可能性はあるが、stable key としては
            #  "tfidf" と別の日本語名なので競合しない).
            _begin_phase("tfidf")
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
            _end_phase()  # U5: tfidf phase 終了

            attach_entities_to_chunks(all_chunks, entities_per_chunk)
            progress.info("エンティティ集計中...")
            # U5: ner phase boundary.
            _begin_phase("ner")
            aggregates, agg_stats = aggregate_entities(
                all_chunks,
                entities_per_chunk,
                min_mentions=self.cfg.min_mentions,
                min_chunks=self.cfg.min_chunks,
            )
            # U5: entities_generated を IngestResult に持ち上げ
            # (現在は log.md にしか出ない AggregationStats.accepted_entities).
            result.entities_generated = agg_stats.accepted_entities
            _end_phase()  # U5: ner phase 終了

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
            # U5: wiki phase boundary. `--skip-wiki` 時も 0 秒で begin/end を
            # 通すことで、stable key set ("wiki": 0.0) を維持する.
            _begin_phase("wiki")
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
                    # F-015: systemic-abort 時は staging の manifest を sibling dir に退避.
                    _preserve_failed_manifest(self.cfg.output_dir, staging)
                    # runtime wiki 失敗 → exit 6. preflight 失敗 (exit 3) と
                    # 区別するため WikiGenerationError にラップする.
                    # U5: phase duration は outermost except → finally で
                    # `_end_phase()` が呼ばれて記録されるため、ここで `_end_phase()`
                    # を呼ばなくても wiki の partial duration は残る.
                    raise WikiGenerationError(
                        f"wiki generation aborted: {e}",
                        reason=type(e).__name__,
                    ) from e

                # 8. index.md (--skip-wiki 時には作らない — F-009)
                _write_index_md(staging / "index.md", staging / "entities" / "manifest.json")
            _end_phase()  # U5: wiki phase 終了 (skip_wiki=True でも記録)

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

            # U4: skipped_files.jsonl (skip 0 件なら作成しない).
            # staging に書き出して atomic swap と一緒に公開する.
            if result.skips:
                _write_skipped_files_jsonl(
                    staging / "skipped_files.jsonl", result.skips
                )

            # 10. atomic swap
            # U5: swap phase boundary.
            _begin_phase("swap")
            _atomic_swap(self.cfg.output_dir, staging)
            _end_phase()
            result.chunks_generated = len(all_chunks)
            result.wiki_stats = wiki_stats
            # U5: 最終段で phase_timings を result に写す.
            result.phase_timings = progress.phase_timings()
            return result
        except LorebookError as e:
            # U1: typed LorebookError は e.exit_code を respect. context と
            # class 名は to_jsonable() で machine-readable に記録する.
            # U5: 失敗 phase の partial duration を記録する (try/finally 相当).
            _end_phase()
            result.exit_code = e.exit_code
            result.errors.append(e.to_jsonable())
            result.phase_timings = progress.phase_timings()
            return result
        except Exception as e:
            # U1: 未分類例外は exit 10 fallback. class 名だけは残す.
            # U5: 失敗 phase の partial duration を記録する.
            _end_phase()
            logger.exception("ingest failed")
            result.exit_code = 10
            result.errors.append(
                {
                    "class": type(e).__name__,
                    "message": f"ingest failure: {e}",
                    "context": {},
                }
            )
            result.phase_timings = progress.phase_timings()
            return result
        finally:
            # F-005/F-058: 早期 return (exit_code 5/6/10 等) 時にも staging をクリーンアップ.
            # atomic swap が成功した場合は staging は既に rename 済みで存在しない.
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


# ---- helpers --------------------------------------------------------


# U4: EncodingError.context の reason 値のうち「auto 検出の失敗」系を
# encoding_detection_failed に分類、それ以外は encoding_decode_failed 扱い.
_DETECTION_FAILURE_REASONS: frozenset[str] = frozenset(
    {
        "detection_ambiguous",
        "too_short_to_detect",
        "utf8_failed_detector_unavailable",
    }
)


def _safe_stat_size(path: Path) -> int | None:
    """`path.stat().st_size` を best-effort で取得. OSError なら None."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _skip_report_from_encoding_error(
    path: Path, exc: EncodingError
) -> SkipReport:
    """`EncodingError` を SkipReport に translate する (U3 の context を活用)."""
    context = exc.context or {}
    ctx_reason = context.get("reason")
    reason = (
        "encoding_detection_failed"
        if ctx_reason in _DETECTION_FAILURE_REASONS
        else "encoding_decode_failed"
    )
    encoding_attempted = (
        context.get("encoding_attempted")
        or context.get("encoding")
        or context.get("tried")
    )
    size_bytes = context.get("size_bytes")
    if size_bytes is None:
        size_bytes = _safe_stat_size(path)
    return SkipReport(
        path=str(path),
        reason=reason,
        detail=str(exc),
        encoding_attempted=encoding_attempted,
        size_bytes=size_bytes,
    )


def _write_skipped_files_jsonl(path: Path, skips: list[SkipReport]) -> None:
    """U4: skipped_files.jsonl を staging dir 内に書き出す.

    `skips` が空なら呼び出し側で skip されるべきだが、safety net として空でも
    書き出さない (出力 dir を clean に保つ plan Approach).
    """
    if not skips:
        return
    with path.open("w", encoding="utf-8") as f:
        for sr in skips:
            f.write(json.dumps(sr.to_jsonable(), ensure_ascii=False))
            f.write("\n")


def _validate_encoding_option(encoding: str) -> None:
    """U3: `--encoding` の妥当性を早期チェック.

    - `"auto"` は detect_encoding の staged pipeline を意味するため pass.
    - それ以外は `codecs.lookup()` で Python codec 名として解決可能であるこ
      とを要求. 解決不能なら `ConfigError(reason="invalid_encoding")` (exit 2).
    """
    if encoding == "auto":
        return
    import codecs as _codecs
    try:
        _codecs.lookup(encoding)
    except LookupError as exc:
        raise ConfigError(
            f"invalid encoding name: {encoding!r}",
            reason="invalid_encoding",
            encoding=encoding,
        ) from exc


def _validate_glob_patterns(globs: tuple[str, ...]) -> None:
    """U2: glob pattern の前提 (non-empty / relative / no path separator 開始) を検証.

    違反時は `ConfigError(reason="invalid_glob")` を raise. recursive discovery でも
    `rglob` に絶対パスや path-separator 開始は渡せないため、ここで早期に弾く.
    """
    if not globs:
        raise ConfigError(
            "at least one --glob pattern is required",
            reason="invalid_glob",
            pattern="",
        )
    for pattern in globs:
        if not pattern:
            raise ConfigError(
                "empty --glob pattern is not allowed",
                reason="invalid_glob",
                pattern=pattern,
            )
        p = Path(pattern)
        if p.is_absolute() or pattern.startswith(("/", os.sep)):
            raise ConfigError(
                f"absolute --glob pattern is not allowed: {pattern!r}",
                reason="invalid_glob",
                pattern=pattern,
            )


def _collect_input_files(
    input_dir: Path,
    *,
    globs: tuple[str, ...] = ("*.txt",),
    recursive: bool = False,
) -> list[Path]:
    """U2: recursive/pattern 対応の入力ファイル収集.

    - `recursive=True` なら `rglob`、False なら `glob`
    - 各 pattern を iterate しつつ `resolve()` 済みパスで dedupe
    - 昇順 sort で決定論的順序を返す
    - pattern 妥当性は caller からも使えるよう `_validate_glob_patterns` で検証
    """
    _validate_glob_patterns(globs)
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    seen: dict[Path, Path] = {}
    for pattern in globs:
        iterator = (
            input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        )
        for path in iterator:
            if not path.is_file():
                continue
            key = path.resolve()
            if key not in seen:
                seen[key] = path
    return sorted(seen.values())


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

    U1: 既存 OSError の copytree fallback 分岐は保持するが、ここを貫通する
    OSError (permission / disk full / 例外的な EXDEV fallback 失敗) は
    AtomicSwapError (exit 16) にラップする. U6 がこの関数を抜本的に書き換える
    予定のため、U1 はこの外殻の translation layer のみ提供する.
    """
    try:
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
    except OSError as e:
        raise AtomicSwapError(
            f"atomic swap failed: {e}",
            reason=type(e).__name__,
            errno=getattr(e, "errno", None) or 0,
            source=str(staging),
            target=str(target),
        ) from e


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


def _parse_globs_arg(value: str | None) -> tuple[str, ...]:
    """U2: `--glob` のカンマ区切り文字列 → tuple.

    空白は trim し、空要素は落とす. 全部空だった場合は空 tuple を返して
    後段の `_validate_glob_patterns` に任せる (ConfigError(reason="invalid_glob")).
    """
    if value is None:
        return ("*.txt",)
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    return parts


def _build_analyzer_meta(output_dir: Path) -> dict[str, Any]:
    """`output_dir/analyzer.json` から RunReport.analyzer フィールドを構築する.

    ingest 失敗で analyzer.json が書き出されなかった場合 (analyzer_init 失敗
    / chunking ZeroChunks 前に swap 未実行) は既定値で埋める. schema key set
    は常に同じ 6 フィールドを emit して stable にする.

    Fields:
      model_name / model_version / model_sha256 / spacy_version / ginza_version / sudachi_dict
    """
    ap = output_dir / "analyzer.json"
    meta: dict[str, Any] = {
        "model_name": "",
        "model_version": "",
        "model_sha256": "",
        "spacy_version": "",
        "ginza_version": "",
        "sudachi_dict": "",
    }
    if not ap.exists():
        return meta
    try:
        data = json.loads(ap.read_text(encoding="utf-8"))
        from lorebook_chunker.wiki import compute_analyzer_json_hash

        strict = data.get("strict_match") or {}
        compat = data.get("compat_match") or {}
        meta["model_name"] = str(strict.get("model_name", ""))
        # Ginza 系モデルでは model_version は ginza package version と同一.
        meta["model_version"] = str(compat.get("ginza", ""))
        try:
            meta["model_sha256"] = compute_analyzer_json_hash(ap)
        except Exception:  # pragma: no cover
            meta["model_sha256"] = ""
        meta["spacy_version"] = str(compat.get("spacy", ""))
        meta["ginza_version"] = str(compat.get("ginza", ""))
        dict_pkg = compat.get("sudachidict_package", "")
        dict_ver = compat.get("sudachidict_package_version", "")
        if dict_pkg and dict_ver:
            meta["sudachi_dict"] = f"{dict_pkg} {dict_ver}"
        elif dict_pkg:
            meta["sudachi_dict"] = str(dict_pkg)
        elif dict_ver:
            meta["sudachi_dict"] = str(dict_ver)
    except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupt / io
        pass
    return meta


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
        # U2: 入力探索オプション.
        recursive=getattr(args, "recursive", False),
        globs=_parse_globs_arg(getattr(args, "globs", None)),
        # U3: encoding. 値のバリデーションは cli._validate_encoding_arg で
        # 事前チェック済み. ここでは素通しで IngestConfig に載せる.
        encoding=getattr(args, "encoding", "auto"),
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

    # U5: started_at / completed_at / duration_seconds を run_report に
    # 記録するため、IngestRunner の前後で wall-clock + monotonic を読む.
    started_at_dt = datetime.now(timezone.utc).astimezone()
    t0 = time.perf_counter()
    result = IngestRunner(
        cfg,
        analyzer_factory=_factory,
        llm_factory=get_client,
    ).run()
    duration_seconds = time.perf_counter() - t0
    completed_at_dt = datetime.now(timezone.utc).astimezone()
    for w in result.warnings:
        print(f"[warn] {w}", file=sys.stderr)
    for e in result.errors:
        print(f"[error] {e}", file=sys.stderr)

    # U5: run_report.json を成功・失敗を問わず常に emit.
    # `IngestResult.errors` は `{class, message, context}` dict なので
    # exit_reason にそのまま使える.
    from lorebook_chunker.run_report import RunReport, write as _write_run_report

    exit_reason: dict[str, Any] | None = None
    if result.exit_code != 0 and result.errors:
        exit_reason = dict(result.errors[0])

    wiki_stats = result.wiki_stats
    llm_model_id = (wiki_stats.llm_model_id if wiki_stats else "") or ""
    if not llm_model_id and cfg.skip_wiki:
        llm_model_id = "noop@skip-wiki"
    wiki_pages_written = wiki_stats.succeeded if wiki_stats else 0

    report = RunReport(
        exit_code=result.exit_code,
        exit_reason=exit_reason,
        started_at=started_at_dt.isoformat(),
        completed_at=completed_at_dt.isoformat(),
        duration_seconds=duration_seconds,
        phase_durations_seconds=dict(result.phase_timings),
        input={
            "input_dir": str(cfg.input_dir),
            "recursive": bool(cfg.recursive),
            "globs": list(cfg.globs),
            "encoding_option": cfg.encoding,
            "files_processed": result.total_input_files - result.skipped_files,
            "files_skipped": [sr.to_jsonable() for sr in result.skips],
        },
        output={
            "output_dir": str(cfg.output_dir),
            "chunks_generated": result.chunks_generated,
            "entities_generated": result.entities_generated,
            "wiki_pages_written": int(wiki_pages_written),
        },
        analyzer=_build_analyzer_meta(cfg.output_dir),
        llm={
            "backend": cfg.llm_backend,
            "model_id": llm_model_id,
            "total_input_tokens": wiki_stats.total_input_tokens if wiki_stats else 0,
            "total_output_tokens": wiki_stats.total_output_tokens if wiki_stats else 0,
        },
        warnings=list(result.warnings),
    )
    report_dict = report.to_json_dict()
    # Writer はエラー時も含めて常に試みる. 失敗なら RunReportError (exit 17) に
    # 置換するが、既に非 0 exit で終わっている場合は元の exit_code を優先する
    # (writer 失敗が一次失敗を mask しないため).
    try:
        _write_run_report(cfg.output_dir / "run_report.json", report)
    except Exception as write_err:  # RunReportError 含む
        if result.exit_code == 0:
            # 成功 run で writer だけが失敗した場合 → exit 17 に昇格.
            from lorebook_chunker.errors import RunReportError

            if not isinstance(write_err, RunReportError):
                write_err = RunReportError(
                    f"run_report.json write failed: {write_err}",
                    path=str(cfg.output_dir / "run_report.json"),
                    reason=type(write_err).__name__,
                )
            result.exit_code = write_err.exit_code
            result.errors.append(write_err.to_jsonable())
            print(f"[error] {write_err}", file=sys.stderr)
        else:
            # 既に失敗している run は元の exit_code を維持し、writer 失敗は
            # warning として stderr にだけ残す.
            logger.warning("failed to write run_report.json: %s", write_err)

    fmt = getattr(args, "format", "human")
    if result.exit_code == 0:
        if fmt == "json":
            print(json.dumps(report_dict, ensure_ascii=False))
        elif not quiet:
            print(
                f"ingest OK: {result.chunks_generated} chunks from "
                f"{result.total_input_files - result.skipped_files} files -> {cfg.output_dir}",
                file=sys.stderr,
            )
    else:
        if fmt == "json":
            print(json.dumps(report_dict, ensure_ascii=False))
    return result.exit_code
