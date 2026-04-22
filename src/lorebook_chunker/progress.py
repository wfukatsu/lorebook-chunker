"""ProgressReporter: ingest の長時間 phase に進捗を出すための簡易レポーター.

stdlib のみで書く (tqdm 非依存). stderr にレート制限 (~5 Hz) で re-draw し、
TTY なら ``\r`` で 1 行を更新、非 TTY なら改行付きで書き出す.

設計要点:
- ``enabled=False`` なら全メソッドが no-op.
- nlp.pipe のようなジェネレータは ``wrap_iter`` で包めば自動で tick される.
- 同時に複数 phase を追跡することは想定しない (ingest の順次パイプライン前提).
"""
from __future__ import annotations

import sys
import time
from typing import IO, Iterable, Iterator, TypeVar

T = TypeVar("T")


class ProgressReporter:
    """ingest の phase 進捗を stderr に書き出す軽量レポーター."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        stream: IO[str] | None = None,
        min_redraw_interval_s: float = 0.2,
    ) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._min_redraw = min_redraw_interval_s
        self._phase: str | None = None
        self._phase_start: float = 0.0
        self._total: int = 0
        self._count: int = 0
        self._last_redraw: float = 0.0
        self._needs_final_newline: bool = False

    # ---- one-shot log ---------------------------------------------------

    def info(self, message: str) -> None:
        """ワンショットの情報行 (phase を持たない)."""
        if not self.enabled:
            return
        self._finish_inline()
        print(f"[lorebook-chunker] {message}", file=self.stream, flush=True)

    # ---- phase lifecycle -------------------------------------------------

    def start(self, name: str, *, total: int | None = None) -> None:
        """phase を開始. total (既知なら) を渡すと進捗率を計算する."""
        if not self.enabled:
            return
        self._finish_inline()
        self._phase = name
        self._phase_start = time.perf_counter()
        self._total = int(total) if total is not None else 0
        self._count = 0
        self._last_redraw = 0.0
        header = f"[lorebook-chunker] {name} 開始"
        if self._total:
            header += f" (総 {self._total})"
        print(header, file=self.stream, flush=True)

    def tick(self, inc: int = 1) -> None:
        """phase の進捗を inc 進める. レート制限付きで inline 再描画."""
        if not self.enabled or self._phase is None:
            return
        self._count += inc
        now = time.perf_counter()
        if now - self._last_redraw < self._min_redraw:
            return
        self._last_redraw = now
        self._redraw(now)

    def end(self) -> None:
        """phase を完了させ、合計時間と処理件数を書き出す."""
        if not self.enabled or self._phase is None:
            return
        elapsed = time.perf_counter() - self._phase_start
        self._finish_inline()
        rate = self._count / elapsed if elapsed > 0 else 0.0
        total_note = f"/{self._total}" if self._total else ""
        print(
            f"[lorebook-chunker] {self._phase} 完了 "
            f"{self._count}{total_note} items in {elapsed:.1f}s ({rate:.1f}/s)",
            file=self.stream,
            flush=True,
        )
        self._phase = None
        self._count = 0
        self._total = 0

    # ---- generator wrapper ---------------------------------------------

    def wrap_iter(self, it: Iterable[T], *, inc: int = 1) -> Iterator[T]:
        """iterator を包んで各 yield 時に tick する."""
        if not self.enabled:
            yield from it
            return
        for item in it:
            yield item
            self.tick(inc)

    # ---- internals ------------------------------------------------------

    def _redraw(self, now: float) -> None:
        elapsed = now - self._phase_start
        rate = self._count / elapsed if elapsed > 0 else 0.0
        if self._total > 0:
            pct = 100.0 * self._count / self._total
            eta = (self._total - self._count) / rate if rate > 0 else 0.0
            body = (
                f"  {self._phase}: {self._count}/{self._total} "
                f"({pct:5.1f}%, {rate:.1f}/s, ETA {eta:5.0f}s, elapsed {elapsed:.0f}s)"
            )
        else:
            body = f"  {self._phase}: {self._count} ({rate:.1f}/s, elapsed {elapsed:.0f}s)"

        if self._is_tty:
            # Clear to end-of-line + carriage return for single-line update
            print("\r\x1b[2K" + body, end="", file=self.stream, flush=True)
            self._needs_final_newline = True
        else:
            # non-TTY: emit a full line each redraw (harmless for logs)
            print(body, file=self.stream, flush=True)
            self._needs_final_newline = False

    def _finish_inline(self) -> None:
        """inline (TTY) の redraw 行があれば改行で閉じる."""
        if self._needs_final_newline and self._is_tty:
            print("", file=self.stream, flush=True)
        self._needs_final_newline = False


__all__ = ["ProgressReporter"]
