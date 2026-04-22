"""Entity wiki generator: source_hash + manifest + pre-flight + retry/budget/systemic failure.

plan の Unit 7 に対応。コンポーネント:
- compute_source_hash / normalize_chunks_for_hash
- ManifestStore (atomic read/write)
- WikiGenerator.generate_all orchestration
"""
from __future__ import annotations

import hashlib
import importlib.resources as pkg_resources
import json
import logging
import os
import random
import tempfile
import threading
import time
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import yaml

from lorebook_chunker.llm import (
    GenerateResult,
    LLMClient,
    LLMPermanentError,
    LLMPreflightError,
    LLMRetryableError,
)
from lorebook_chunker.ner import sanitize_entity_filename
from lorebook_chunker.schema import ChunkRecord, EntityAggregate

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_VERSION = "v1"
PROMPT_TEMPLATE_V1 = """以下のテキスト断片から、{ner_label} の「{entity_name}」について 2〜4 文で日本語の要約を書いてください。
テキストに書かれていない事実を想像で補わないでください。該当テキストで確認できる内容のみを使ってください。

## 関連テキスト断片
{chunks_text}

## 出力形式
2〜4 文の日本語要約のみ。前置きや見出しは不要。
"""

SUMMARY_BODY_MARKER_BEGIN = "<!-- summary-begin -->"
SUMMARY_BODY_MARKER_END = "<!-- summary-end -->"

PREFLIGHT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 400
DEFAULT_RETRY_ATTEMPTS = 3
SYSTEMIC_FAILURE_MIN_ATTEMPTS = 5
SYSTEMIC_FAILURE_RATIO = 0.5
MANIFEST_CHECKPOINT_INTERVAL = 5
# エンティティプロンプトに含めるチャンク上限.
# 頻出エンティティ (100+ チャンクに出現) のプロンプトを無制限に積むと入力トークンが
# 爆発する. 先頭 N チャンクに絞ることで上限をかけつつ、`EntityAggregate.chunk_ids`
# は登場順でソート済みなので情報量は冒頭に集中する.
DEFAULT_PROMPT_CHUNK_CAP = 20

EntityStatus = Literal["success", "failed", "budget_skipped"]

# LLM 出力の中に裸の `---` があるとフロントマター区切りを割る (F-013)。
_YAML_FRONTMATTER_FENCE = re.compile(r"^---$", re.MULTILINE)


def _fence_yaml_markers(body: str) -> str:
    """LLM 応答内の ``^---$`` を ``\\---`` にエスケープして YAML frontmatter を守る."""
    return _YAML_FRONTMATTER_FENCE.sub(r"\\---", body)


# ---- source_hash --------------------------------------------------------

def _canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _normalize_chunks_for_hash(chunks: Sequence[ChunkRecord], entity_chunk_ids: set[str]) -> str:
    """エンティティが出現するチャンクテキストを chunk_id 昇順で sorted 連結.

    ChunkRecord.text はすでに ingest 側で ``normalize_text`` 済み (chunker は
    正規化後テキストから切り出すのみ) なので、ここで再正規化しても冪等な no-op になる.
    純粋関数の高速化のため再正規化を省略する.
    """
    picked = [c for c in chunks if c.chunk_id in entity_chunk_ids]
    picked.sort(key=lambda c: c.chunk_id)
    return "\n---\n".join(c.text for c in picked)


def compute_source_hash(
    *,
    entity_name: str,
    ner_label: str,
    chunks: Sequence[ChunkRecord],
    entity_chunk_ids: set[str],
    analyzer_json_hash: str,
    ginza_model_version: str,
    llm_model_id: str,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> str:
    parts = [
        entity_name,
        ner_label,
        _normalize_chunks_for_hash(chunks, entity_chunk_ids),
        analyzer_json_hash,
        ginza_model_version,
        llm_model_id,
        prompt_template_version,
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def compute_analyzer_json_hash(path: str | Path) -> str:
    """analyzer.json の canonical JSON ハッシュを返す."""
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return _canonical_json_hash(obj)


# ---- Manifest store -----------------------------------------------------

@dataclass
class ManifestEntry:
    entity_name: str
    ner_label: str
    source_hash: str
    status: EntityStatus
    mention_count: int
    chunk_count: int
    chunk_ids: list[str]
    cooccurring_entities: list[dict[str, str]] = field(default_factory=list)
    last_attempt_at: str = ""
    failure_reason: str = ""

    def key(self) -> str:
        return f"{self.ner_label}__{self.entity_name}"


@dataclass
class ManifestStats:
    """manifest 書き出し時の統計."""

    success: int = 0
    failed: int = 0
    budget_skipped: int = 0
    skipped_cached: int = 0


@dataclass
class WikiStats:
    """1 回の ingest の wiki 生成サマリ (log.md に書かれる)."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    budget_skipped: int = 0
    skipped_cached: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    preflight_called: bool = False
    preflight_calls: int = 0  # pre-flight が実 LLM 呼び出しを消費した回数
    systemic_failure_aborted: bool = False
    llm_model_id: str = ""


class ManifestStore:
    """entities/manifest.json の atomic read/write."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, ManifestEntry] = {}
        self._header: dict[str, Any] = {
            "version": 1,
            "generated_at": "",
            "ginza_model": "",
            "llm_model": "",
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        }

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """manifest を読み込み. 存在しない or 破損なら空状態に."""
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._header = {
                "version": data.get("version", 1),
                "generated_at": data.get("generated_at", ""),
                "ginza_model": data.get("ginza_model", ""),
                "llm_model": data.get("llm_model", ""),
                "prompt_template_version": data.get(
                    "prompt_template_version", PROMPT_TEMPLATE_VERSION
                ),
            }
            entries = data.get("entries", {}) or {}
            for key, raw in entries.items():
                self._entries[key] = ManifestEntry(**raw)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("manifest.json corrupted (%s); starting with empty state", e)
            self._entries = {}

    def get(self, key: str) -> ManifestEntry | None:
        return self._entries.get(key)

    def set(self, entry: ManifestEntry) -> None:
        self._entries[entry.key()] = entry

    def entries(self) -> list[ManifestEntry]:
        return list(self._entries.values())

    def save(
        self,
        *,
        ginza_model: str,
        llm_model: str,
    ) -> None:
        """atomic write: same-fs tempfile + fsync + os.replace + parent dir fsync."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._header["version"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ginza_model": ginza_model,
            "llm_model": llm_model,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "entries": {k: asdict(v) for k, v in self._entries.items()},
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".manifest.", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        # 親ディレクトリの dirent も同期 (POSIX only)
        try:
            dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, NotImplementedError):  # Windows など
            pass


# ---- Wiki page writer ---------------------------------------------------

def _render_wiki_page(
    entry: ManifestEntry,
    summary_body: str,
    *,
    ginza_model: str,
    llm_model: str,
    ai_verification_status: str,
    chunk_snippets: list[dict[str, Any]],
) -> str:
    frontmatter: dict[str, Any] = {
        "entity_name": entry.entity_name,
        "ner_label": entry.ner_label,
        "mention_count": entry.mention_count,
        "chunk_count": entry.chunk_count,
        "source_hash": entry.source_hash,
        "ginza_model": ginza_model,
        "llm_model": llm_model,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "ai_verification_status": ai_verification_status,
        "cooccurring_entities": entry.cooccurring_entities,
        # F-006: plan R15 で必須の 3 フィールド (manifest のコピー)
        "status": entry.status,
        "last_attempt_at": entry.last_attempt_at,
        "chunk_ids": list(entry.chunk_ids),
    }
    fm_text = yaml.safe_dump(
        frontmatter,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).rstrip()

    lines: list[str] = []
    lines.append("---")
    lines.append(fm_text)
    lines.append("---")
    lines.append("")
    lines.append(f"# {entry.entity_name} ({entry.ner_label})")
    lines.append("")
    lines.append(SUMMARY_BODY_MARKER_BEGIN)
    # F-013: 裸の `---` 行は YAML frontmatter を壊すのでエスケープする.
    lines.append(_fence_yaml_markers(summary_body.strip()))
    lines.append(SUMMARY_BODY_MARKER_END)
    lines.append("")
    lines.append("## 出現チャンク")
    lines.append("")
    for snippet in chunk_snippets:
        lines.append(f"- `{snippet['chunk_id']}` ({snippet['source']})")
        excerpt = snippet.get("excerpt", "").strip()
        if excerpt:
            lines.append(f"  > {excerpt}")
    lines.append("")
    return "\n".join(lines)


def _build_chunk_snippets(
    chunks_by_id: dict[str, ChunkRecord], chunk_ids: Iterable[str], span: int = 40
) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    for cid in chunk_ids:
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        # 先頭の一部だけを excerpt として付ける (エンティティ位置を使うバリエーションは後続で)
        excerpt = chunk.text[: span * 2]
        snippets.append(
            {
                "chunk_id": cid,
                "source": chunk.source,
                "excerpt": excerpt.replace("\n", " "),
            }
        )
    return snippets


# ---- Wiki generator -----------------------------------------------------

@dataclass
class WikiGeneratorConfig:
    entities_dir: Path
    manifest_path: Path
    max_tokens_per_entity: int = DEFAULT_MAX_TOKENS
    max_llm_calls: int | None = None
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    skip_wiki: bool = False
    force_regenerate: bool = False
    retry_failed: bool = False
    ai_verification_status: str = "unverified"
    # parallelism=1 (default) でシリアル実行. >1 で ThreadPoolExecutor を使い
    # I/O-bound な LLM 呼び出しを同時投入する.
    # - Anthropic: 5〜10 が目安 (公式 concurrency limit 内)
    # - Ollama: OLLAMA_NUM_PARALLEL と揃える (通常 3〜4, M2 Pro 32GB qwen3:8b なら 3)
    parallelism: int = 1
    # LLM プロンプトに載せるチャンク上限. None で無制限. 既定は DEFAULT_PROMPT_CHUNK_CAP.
    prompt_chunk_cap: int | None = DEFAULT_PROMPT_CHUNK_CAP
    # progress レポーター (任意). IngestRunner 側で注入.
    progress: Any = None


class WikiGenerator:
    """エンティティ wiki 生成のオーケストレーション."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        config: WikiGeneratorConfig,
        analyzer_json_hash: str,
        ginza_model_version: str,
        chunks: Sequence[ChunkRecord],
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._llm = llm_client
        self._config = config
        self._analyzer_json_hash = analyzer_json_hash
        self._ginza_model_version = ginza_model_version
        self._chunks = list(chunks)
        # F-042: 全メソッドで共有する chunk_id -> ChunkRecord index.
        self._chunks_by_id = {c.chunk_id: c for c in self._chunks}
        self._manifest = ManifestStore(config.manifest_path)
        self._sleep = sleep
        self._rng = rng or random.Random(42)
        # parallel 実行時の manifest / page 書き込み排他. parallelism=1 でもロック取得
        # 自体のコストは無視できるので常時使う.
        self._write_lock = threading.Lock()

    def generate_all(self, entities: Sequence[EntityAggregate]) -> WikiStats:
        stats = WikiStats(llm_model_id=self._llm.model_id)
        if self._config.skip_wiki:
            # 既存 manifest を保存 (空 or 既存のまま) して early return
            self._manifest.load()
            self._manifest.save(
                ginza_model=self._ginza_model_version, llm_model=self._llm.model_id
            )
            return stats
        if not entities:
            self._manifest.save(
                ginza_model=self._ginza_model_version, llm_model=self._llm.model_id
            )
            return stats

        # 既存 manifest を読む
        self._manifest.load()

        # pre-flight (config で実呼び出しなら skip するオプションもあり得るが v1 は常時実行)
        self._preflight()
        stats.preflight_called = True
        stats.preflight_calls += 1

        # 優先順序
        ordered = sorted(
            entities,
            key=lambda e: (-e.mention_count, -e.chunk_count, e.name),
        )

        # まず cache/budget で処理できるものを sync に捌いてから、残りを LLM に投げる.
        # 残り (= 実際に LLM 呼び出しが発生する) だけを submit することで、budget と
        # skipped_cached は parallelism 有無に関係なく deterministic に決まる.
        to_submit: list[tuple[EntityAggregate, str]] = []
        for agg in ordered:
            key = f"{agg.ner_label}__{agg.name}"
            source_hash = compute_source_hash(
                entity_name=agg.name,
                ner_label=agg.ner_label,
                chunks=self._chunks,
                entity_chunk_ids=set(agg.chunk_ids),
                analyzer_json_hash=self._analyzer_json_hash,
                ginza_model_version=self._ginza_model_version,
                llm_model_id=self._llm.model_id,
            )
            prior = self._manifest.get(key)
            if self._should_skip_due_to_cache(prior, source_hash):
                stats.skipped_cached += 1
                continue
            # F-003: budget はエンティティ試行数のみでカウント.
            # pre-flight は別カウンタ (stats.preflight_calls) で記録し、
            # --max-llm-calls の budget からは外す.
            if (
                self._config.max_llm_calls is not None
                and len(to_submit) >= self._config.max_llm_calls
            ):
                self._record_budget_skip(agg)
                stats.budget_skipped += 1
                continue
            to_submit.append((agg, source_hash))

        parallelism = max(1, int(self._config.parallelism or 1))
        progress = self._config.progress
        if progress is not None and to_submit:
            progress.start(
                f"wiki 生成 (LLM 呼出 parallelism={parallelism})", total=len(to_submit)
            )

        try:
            if parallelism == 1:
                self._run_serial(to_submit, stats)
            else:
                self._run_parallel(to_submit, stats, parallelism)
        finally:
            if progress is not None and to_submit:
                progress.end()

        # 最終 manifest 書き出し
        self._manifest.save(
            ginza_model=self._ginza_model_version, llm_model=self._llm.model_id
        )
        return stats

    # ---- serial / parallel entity loops ------------------------------

    def _run_serial(
        self,
        to_submit: list[tuple[EntityAggregate, str]],
        stats: WikiStats,
    ) -> None:
        """parallelism=1 用の逐次ループ (旧挙動をそのまま保つ).

        systemic abort / checkpoint / 成功失敗記録の契約はこのパスが基準.
        """
        progress = self._config.progress
        attempts_completed = 0
        for agg, source_hash in to_submit:
            stats.attempted += 1
            result = self._generate_for_entity(agg)
            self._apply_result(agg, source_hash, result, stats)
            attempts_completed += 1
            if progress is not None:
                progress.tick()
            if attempts_completed % MANIFEST_CHECKPOINT_INTERVAL == 0:
                self._manifest.save(
                    ginza_model=self._ginza_model_version, llm_model=self._llm.model_id
                )
            if self._should_abort_systemic(stats):
                self._handle_systemic_abort(stats)

    def _run_parallel(
        self,
        to_submit: list[tuple[EntityAggregate, str]],
        stats: WikiStats,
        parallelism: int,
    ) -> None:
        """ThreadPoolExecutor を使った同時 LLM 呼び出し.

        - budget / cache は pre-compute (to_submit) で確定済み.
        - 完了ごとに manifest checkpoint + systemic abort を判定し、検知したら
          pool を shutdown(cancel_futures=True) して LLMPermanentError を raise.
        - 完了順は確定しないが manifest への書き込みは内部ロックで直列化されるため
          出力 .md / manifest.json は決定論.
        """
        progress = self._config.progress
        attempts_completed = 0
        pool = ThreadPoolExecutor(
            max_workers=parallelism, thread_name_prefix="wiki-llm"
        )
        # submit 前に attempted を加算しておくと systemic check が
        # "in-flight で投入済みだが未完了" のものも含めてしまう.
        # attempted は完了ごとに加算する (逐次版と意味を合わせる).
        futures: dict[Future[GenerateResult | None], tuple[EntityAggregate, str]] = {}
        try:
            for agg, source_hash in to_submit:
                fut = pool.submit(self._generate_for_entity, agg)
                futures[fut] = (agg, source_hash)
            # as_completed の代わりに毎回スキャンする: systemic abort 時に
            # cancel_futures=True で shutdown してから loop を抜けるため.
            from concurrent.futures import as_completed

            for fut in as_completed(futures):
                agg, source_hash = futures[fut]
                try:
                    result = fut.result()
                except BaseException as e:  # pragma: no cover - defensive
                    logger.exception("unexpected exception from wiki worker: %s", e)
                    result = None
                stats.attempted += 1
                self._apply_result(agg, source_hash, result, stats)
                attempts_completed += 1
                if progress is not None:
                    progress.tick()
                if attempts_completed % MANIFEST_CHECKPOINT_INTERVAL == 0:
                    self._manifest.save(
                        ginza_model=self._ginza_model_version,
                        llm_model=self._llm.model_id,
                    )
                if self._should_abort_systemic(stats):
                    # 未完了は budget_skipped 相当として記録.
                    # (source_hash 未確定なので record_budget_skip の既存 prior からの抽出を使う.)
                    for pending, (pend_agg, _pend_hash) in list(futures.items()):
                        if not pending.done():
                            pending.cancel()
                    self._handle_systemic_abort(stats)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    # ---- shared helpers for serial/parallel --------------------------

    def _apply_result(
        self,
        agg: EntityAggregate,
        source_hash: str,
        result: GenerateResult | None,
        stats: WikiStats,
    ) -> None:
        """LLM 呼び出し結果を stats と manifest/page に反映 (シリアル/並列で共通).

        manifest/page 書き込みは `self._write_lock` で直列化し、parallel 実行時も
        出力ファイルは決定論的に置かれることを保証する.
        """
        with self._write_lock:
            if result is None:
                stats.failed += 1
                self._record_failure(agg, source_hash, reason="retries exhausted")
            else:
                stats.succeeded += 1
                stats.total_input_tokens += result.input_tokens
                stats.total_output_tokens += result.output_tokens
                self._record_success(agg, source_hash, result)

    def _handle_systemic_abort(self, stats: WikiStats) -> None:
        """F-015 systemic failure abort: manifest を退避保存してから raise."""
        stats.systemic_failure_aborted = True
        self._manifest.save(
            ginza_model=self._ginza_model_version, llm_model=self._llm.model_id
        )
        raise LLMPermanentError(
            f"systemic failure detected: {stats.failed}/{stats.attempted} "
            "entities failed (> 50%). aborting ingest."
        )

    # ---- pre-flight --------------------------------------------------

    def _preflight(self) -> None:
        prompt = pkg_resources.files("lorebook_chunker.resources").joinpath(
            "preflight_prompt.txt"
        ).read_text(encoding="utf-8")
        try:
            result = self._llm.generate(
                prompt=prompt, max_tokens=self._config.max_tokens_per_entity
            )
        except Exception as e:
            raise LLMPreflightError(f"pre-flight failed: {e}") from e
        if not self._is_preflight_success(result):
            raise LLMPreflightError(
                f"pre-flight response did not meet success criteria: "
                f"text_len={len(result.text.strip())} output_tokens={result.output_tokens} "
                f"finish_reason={result.finish_reason!r}"
            )

    @staticmethod
    def _is_preflight_success(result: GenerateResult) -> bool:
        return (
            len(result.text.strip()) > 0
            and result.output_tokens > 0
            and result.finish_reason in ("end_turn", "stop")
        )

    # ---- cache logic -------------------------------------------------

    def _should_skip_due_to_cache(
        self, prior: ManifestEntry | None, current_hash: str
    ) -> bool:
        if self._config.force_regenerate:
            return False
        if prior is None:
            return False
        if prior.source_hash != current_hash:
            return False
        # source_hash 一致: 前回の status で分岐
        if prior.status == "success":
            return True
        if prior.status == "failed":
            # `--retry-failed` 指定時は再試行 (skip しない)
            return not self._config.retry_failed
        if prior.status == "budget_skipped":
            # budget_skipped は `--retry-failed` で明示的に再試行
            return not self._config.retry_failed
        return False

    # ---- per-entity generation --------------------------------------

    def _generate_for_entity(self, agg: EntityAggregate) -> GenerateResult | None:
        prompt = self._build_entity_prompt(agg)
        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                return self._llm.generate(
                    prompt=prompt, max_tokens=self._config.max_tokens_per_entity
                )
            except LLMRetryableError as e:
                if attempt >= self._config.retry_attempts:
                    logger.warning(
                        "entity %s/%s exhausted retries: %s",
                        agg.ner_label,
                        agg.name,
                        e,
                    )
                    return None
                backoff = self._exponential_backoff(attempt)
                logger.info(
                    "retryable LLM error for %s (attempt %d/%d), backing off %.1fs",
                    agg.name,
                    attempt,
                    self._config.retry_attempts,
                    backoff,
                )
                self._sleep(backoff)
            except LLMPermanentError as e:
                logger.warning("permanent LLM error for %s: %s", agg.name, e)
                return None
        return None

    def _exponential_backoff(self, attempt: int) -> float:
        base = min(2.0 ** attempt, 30.0)
        jitter = self._rng.uniform(0.0, 1.0)
        return base + jitter

    def _build_entity_prompt(self, agg: EntityAggregate) -> str:
        cap = self._config.prompt_chunk_cap
        chunk_ids = agg.chunk_ids if cap is None else agg.chunk_ids[: max(1, int(cap))]
        pieces: list[str] = []
        for cid in chunk_ids:
            chunk = self._chunks_by_id.get(cid)
            if chunk is None:
                continue
            pieces.append(f"- [{chunk.source}] {chunk.text}")
        if cap is not None and len(agg.chunk_ids) > len(chunk_ids):
            pieces.append(
                f"... (残り {len(agg.chunk_ids) - len(chunk_ids)} 件のチャンクは省略) ..."
            )
        chunks_text = "\n".join(pieces)
        return PROMPT_TEMPLATE_V1.format(
            ner_label=agg.ner_label,
            entity_name=agg.name,
            chunks_text=chunks_text,
        )

    # ---- recording ---------------------------------------------------

    def _record_success(
        self,
        agg: EntityAggregate,
        source_hash: str,
        result: GenerateResult,
    ) -> None:
        entry = ManifestEntry(
            entity_name=agg.name,
            ner_label=agg.ner_label,
            source_hash=source_hash,
            status="success",
            mention_count=agg.mention_count,
            chunk_count=agg.chunk_count,
            chunk_ids=list(agg.chunk_ids),
            cooccurring_entities=list(agg.cooccurring_entities),
            last_attempt_at=datetime.now(timezone.utc).isoformat(),
            failure_reason="",
        )
        self._manifest.set(entry)
        # wiki ページを書き出し
        filename = sanitize_entity_filename(agg.ner_label, agg.name)
        out_path = self._config.entities_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        snippets = _build_chunk_snippets(self._chunks_by_id, agg.chunk_ids)
        page = _render_wiki_page(
            entry,
            summary_body=result.text,
            ginza_model=self._ginza_model_version,
            llm_model=self._llm.model_id,
            ai_verification_status=self._config.ai_verification_status,
            chunk_snippets=snippets,
        )
        out_path.write_text(page, encoding="utf-8")

    def _record_failure(self, agg: EntityAggregate, source_hash: str, *, reason: str) -> None:
        prior = self._manifest.get(f"{agg.ner_label}__{agg.name}")
        entry = ManifestEntry(
            entity_name=agg.name,
            ner_label=agg.ner_label,
            source_hash=source_hash,
            status="failed",
            mention_count=agg.mention_count,
            chunk_count=agg.chunk_count,
            chunk_ids=list(agg.chunk_ids),
            cooccurring_entities=list(agg.cooccurring_entities),
            last_attempt_at=datetime.now(timezone.utc).isoformat(),
            failure_reason=reason,
        )
        self._manifest.set(entry)
        # 失敗時は wiki ファイルを書かないが、過去に成功ファイルがある場合は残す

    def _record_budget_skip(self, agg: EntityAggregate) -> None:
        prior = self._manifest.get(f"{agg.ner_label}__{agg.name}")
        source_hash_prior = prior.source_hash if prior else ""
        entry = ManifestEntry(
            entity_name=agg.name,
            ner_label=agg.ner_label,
            source_hash=source_hash_prior,
            status="budget_skipped",
            mention_count=agg.mention_count,
            chunk_count=agg.chunk_count,
            chunk_ids=list(agg.chunk_ids),
            cooccurring_entities=list(agg.cooccurring_entities),
            last_attempt_at=datetime.now(timezone.utc).isoformat(),
            failure_reason="budget exhausted",
        )
        self._manifest.set(entry)

    # ---- systemic failure -------------------------------------------

    @staticmethod
    def _should_abort_systemic(stats: WikiStats) -> bool:
        attempted = stats.succeeded + stats.failed
        if attempted < SYSTEMIC_FAILURE_MIN_ATTEMPTS:
            return False
        return stats.failed / attempted > SYSTEMIC_FAILURE_RATIO
