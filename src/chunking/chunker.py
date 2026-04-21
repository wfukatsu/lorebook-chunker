"""Chunker: 文境界尊重の char-target チャンキング + overlap.

入力は **正規化済みテキスト** (呼び出し側が normalize_text 1 回適用後) であり、
chunker は再正規化しない。char_start/char_end は正規化後テキストにおけるオフセット。
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Iterator

from chunking.schema import ChunkRecord

logger = logging.getLogger(__name__)

# Sentence splitter 契約: 正規化済みテキスト → 文文字列のイテレータ.
SentenceSplitter = Callable[[str], Iterable[str]]

DEFAULT_TARGET_CHARS = 500
DEFAULT_OVERLAP_CHARS = 100
DEFAULT_MAX_CHUNK_CHARS = 1500

# ソフト分割フォールバックの順序: 読点 → 改行 → その他
SOFT_SPLIT_PRIORITIES: tuple[str, ...] = ("、", "\n", ",", " ")


@dataclass
class ChunkerWarning:
    """チャンク化中のソフトな警告 (log.md に書き出す想定)."""

    kind: str
    detail: str
    char_offset: int


class Chunker:
    """char-target + overlap + soft-split の3層チャンキング."""

    def __init__(
        self,
        splitter: SentenceSplitter,
        *,
        target_chars: int = DEFAULT_TARGET_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be > 0")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if max_chunk_chars < target_chars:
            raise ValueError("max_chunk_chars must be >= target_chars")
        self._splitter = splitter
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.max_chunk_chars = max_chunk_chars
        self.warnings: list[ChunkerWarning] = []

    # ---- public API --------------------------------------------------

    def chunk_document(
        self,
        source_relative_path: str,
        normalized_text: str,
        *,
        input_dir: str | None = None,
    ) -> Iterator[ChunkRecord]:
        """1 ドキュメントをチャンク列に変換."""
        if not normalized_text:
            return
        sentences = list(self._splitter(normalized_text))
        if not sentences:
            return
        sentence_spans = self._locate_sentence_spans(normalized_text, sentences)
        if not sentence_spans:
            return
        base_chunks = list(self._accumulate_chunks(sentence_spans, normalized_text))
        final_chunks = self._apply_overlap(base_chunks, sentence_spans, normalized_text)
        posix_source = self._posix_relative(source_relative_path, input_dir)
        for (start, end, text) in final_chunks:
            cid = self._compute_chunk_id(posix_source, start, end, text)
            # row_index は ingest 時にコーパス全体でグローバルに振り直す
            # (ここでは sentinel -1 を埋めておく)
            yield ChunkRecord(
                chunk_id=cid,
                row_index=-1,
                source=posix_source,
                char_start=start,
                char_end=end,
                text=text,
            )

    # ---- sentence span recovery --------------------------------------

    @staticmethod
    def _locate_sentence_spans(text: str, sentences: Iterable[str]) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        cursor = 0
        for sent in sentences:
            if not sent:
                continue
            idx = text.find(sent, cursor)
            if idx == -1:
                stripped = sent.strip()
                if not stripped:
                    continue
                idx = text.find(stripped, cursor)
                if idx == -1:
                    logger.debug("sentence not located: %r", sent[:30])
                    continue
                sent = stripped
            end = idx + len(sent)
            spans.append((idx, end, sent))
            cursor = end
        return spans

    # ---- chunk accumulation ------------------------------------------

    def _accumulate_chunks(
        self,
        sentence_spans: list[tuple[int, int, str]],
        normalized_text: str,
    ) -> Iterator[tuple[int, int, str]]:
        """文を累積してチャンク境界を決める.

        yield する (start, end, text) は overlap 未適用の raw チャンク.
        text は ``normalized_text[start:end]`` そのもの (文間の空白や連結処理を省略).
        """
        buf_start: int | None = None
        buf_end: int | None = None
        buf_len = 0
        for start, end, sent in sentence_spans:
            sent_len = end - start
            if sent_len > self.max_chunk_chars:
                if buf_start is not None and buf_end is not None:
                    yield (buf_start, buf_end, normalized_text[buf_start:buf_end])
                    buf_start = buf_end = None
                    buf_len = 0
                yield from self._soft_split_sentence(start, end, sent)
                continue
            if sent_len > self.target_chars:
                self.warnings.append(
                    ChunkerWarning(
                        kind="sentence_over_target",
                        detail=f"1 sentence {sent_len} chars > target {self.target_chars}",
                        char_offset=start,
                    )
                )
                if buf_start is not None and buf_end is not None:
                    yield (buf_start, buf_end, normalized_text[buf_start:buf_end])
                    buf_start = buf_end = None
                    buf_len = 0
                yield (start, end, sent)
                continue
            if buf_start is None:
                buf_start = start
            buf_end = end
            buf_len += sent_len
            if buf_len >= self.target_chars:
                yield (buf_start, buf_end, normalized_text[buf_start:buf_end])
                buf_start = buf_end = None
                buf_len = 0
        if buf_start is not None and buf_end is not None:
            yield (buf_start, buf_end, normalized_text[buf_start:buf_end])

    def _soft_split_sentence(
        self, start: int, end: int, sent: str
    ) -> Iterator[tuple[int, int, str]]:
        self.warnings.append(
            ChunkerWarning(
                kind="sentence_soft_split",
                detail=f"sentence {end - start} chars > max {self.max_chunk_chars}, soft-splitting",
                char_offset=start,
            )
        )
        remaining = sent
        base = start
        while len(remaining) > self.max_chunk_chars:
            window = remaining[: self.max_chunk_chars]
            cut = -1
            for sep in SOFT_SPLIT_PRIORITIES:
                idx = window.rfind(sep)
                if idx > 0:
                    cut = idx + len(sep)
                    break
            if cut <= 0:
                cut = self.max_chunk_chars
            piece = remaining[:cut]
            yield (base, base + len(piece), piece)
            base += len(piece)
            remaining = remaining[cut:]
        if remaining:
            yield (base, base + len(remaining), remaining)

    # ---- overlap handling --------------------------------------------

    def _apply_overlap(
        self,
        base_chunks: list[tuple[int, int, str]],
        sentence_spans: list[tuple[int, int, str]],
        normalized_text: str,
    ) -> list[tuple[int, int, str]]:
        """各チャンクの先頭に overlap_chars ぶんだけ前チャンク末尾をコピー (文境界にスナップ).

        R4:
          1. (prev_end - N) 以下で最近接の文境界を探す
          2. 窓 [prev_end - 2N, prev_end - N/2] に境界無ければ overlap 0
          3. 2N 超えまで拡張しない
        """
        if not base_chunks or self.overlap_chars == 0:
            return list(base_chunks)
        boundaries = sorted(
            {span[0] for span in sentence_spans} | {span[1] for span in sentence_spans}
        )
        result: list[tuple[int, int, str]] = []
        for i, (start, end, text) in enumerate(base_chunks):
            if i == 0:
                result.append((start, end, text))
                continue
            prev_end = base_chunks[i - 1][1]
            target = prev_end - self.overlap_chars
            window_min = prev_end - 2 * self.overlap_chars
            window_max = prev_end - self.overlap_chars // 2
            snap = self._find_snapped_boundary(target, window_min, window_max, boundaries)
            if snap is None or snap >= start:
                # overlap を取らない: 前チャンク末尾と現チャンク開始が既に一致か、窓内に境界なし
                self.warnings.append(
                    ChunkerWarning(
                        kind="overlap_skipped",
                        detail=f"no snap boundary in window [{window_min}, {window_max}]",
                        char_offset=start,
                    )
                )
                result.append((start, end, text))
                continue
            # snap 位置から end までを切り出してチャンクテキストを再構成
            new_start = snap
            new_text = normalized_text[new_start:end]
            result.append((new_start, end, new_text))
        return result

    @staticmethod
    def _find_snapped_boundary(
        target: int,
        window_min: int,
        window_max: int,
        boundaries: list[int],
    ) -> int | None:
        best: int | None = None
        for b in boundaries:
            if b > window_max:
                break
            if b < window_min:
                continue
            if b <= target:
                if best is None or b > best:
                    best = b
        return best

    # ---- chunk id helpers --------------------------------------------

    @staticmethod
    def _posix_relative(source: str, input_dir: str | None) -> str:
        if input_dir is None:
            return source
        # Windows のバックスラッシュパスにも対応するため native Path を使い、
        # 最後に .as_posix() で forward slash に統一する。
        try:
            return Path(source).relative_to(Path(input_dir)).as_posix()
        except ValueError:
            return Path(source).as_posix()

    @staticmethod
    def _compute_chunk_id(posix_source: str, start: int, end: int, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = f"{posix_source}|{start}|{end}|{text_hash}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# ---- utility: pure-python sentence splitter for tests -----------------

def simple_japanese_splitter(text: str) -> Iterable[str]:
    """テスト用の素朴な文分割器. 「。」「\n」で分割して空を除外."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ("。", "\n"):
            seg = "".join(buf).strip()
            if seg:
                out.append(seg)
            buf = []
    seg = "".join(buf).strip()
    if seg:
        out.append(seg)
    return out
