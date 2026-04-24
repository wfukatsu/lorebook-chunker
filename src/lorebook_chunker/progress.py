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
        # U5: phase duration accumulation. `end()` 時に
        # `self._durations[self._phase] = elapsed` を記録する.
        # `enabled=False` でも計測できるよう、後述の `end()` 内で
        # `enabled` を条件にしない (redraw のみ enabled でガードする).
        self._durations: dict[str, float] = {}

    # ---- one-shot log ---------------------------------------------------

    def info(self, message: str) -> None:
        """ワンショットの情報行 (phase を持たない)."""
        if not self.enabled:
            return
        self._finish_inline()
        print(f"[lorebook-chunker] {message}", file=self.stream, flush=True)

    # ---- phase lifecycle -------------------------------------------------

    def start(self, name: str, *, total: int | None = None) -> None:
        """phase を開始. total (既知なら) を渡すと進捗率を計算する.

        U5: `enabled=False` でも phase 名と `_phase_start` は記録するため、
        `end()` での duration accumulation が機能する. stderr 出力のみを
        `enabled` でガードする.
        """
        self._finish_inline()
        self._phase = name
        self._phase_start = time.perf_counter()
        self._total = int(total) if total is not None else 0
        self._count = 0
        self._last_redraw = 0.0
        if not self.enabled:
            return
        header = f"[lorebook-chunker] {name} 開始"
        if self._total:
            header += f" (総 {self._total})"
        print(header, file=self.stream, flush=True)

    def tick(self, inc: int = 1) -> None:
        """phase の進捗を inc 進める. レート制限付きで inline 再描画."""
        if self._phase is None:
            return
        self._count += inc
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self._last_redraw < self._min_redraw:
            return
        self._last_redraw = now
        self._redraw(now)

    def end(self) -> None:
        """phase を完了させ、合計時間と処理件数を書き出す.

        U5: `self._durations[self._phase] = elapsed` を記録してから state を
        reset する. 同じ phase 名で `end()` が複数回呼ばれた場合は last-write-wins.
        `enabled=False` でも duration は蓄積する (stderr 出力だけが無効化される).
        """
        if self._phase is None:
            return
        elapsed = time.perf_counter() - self._phase_start
        # U5: 失敗 phase も finally 経由でここに到達するため、例外時も
        # duration が記録される (ProgressReporter 側では例外を握らない).
        self._durations[self._phase] = elapsed
        if self.enabled:
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

    # ---- U5: phase timings snapshot -------------------------------------

    def phase_timings(self) -> dict[str, float]:
        """記録済み phase duration の snapshot を返す (caller が mutate しても内部状態に影響しない)."""
        return dict(self._durations)

    def measure_phase(self, name: str) -> "_PhaseTimerCM":
        """outer phase boundary 用の context manager.

        `start`/`end` ペアと違い、inner `start()`/`end()` を呼んでも上書きされず、
        純粋に `time.perf_counter()` 差分を `self._durations[name]` に記録する.

        IngestRunner.run で 6 phase (analyzer_init/chunking/tfidf/ner/wiki/swap)
        の boundary を with 文で囲むために使用. 例外時も `__exit__` 経由で
        duration が記録されるため、try/finally 相当の挙動を持つ.
        """
        return _PhaseTimerCM(self, name)

    def _record_duration(self, name: str, elapsed: float) -> None:
        """`measure_phase` から呼ばれる duration 記録 hook (test 用 public-ish).

        last-write-wins: 同名で複数回記録されたら最後の値が残る.
        """
        self._durations[name] = elapsed

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


class _PhaseTimerCM:
    """`ProgressReporter.measure_phase` の context manager 実装.

    `__exit__` で必ず duration を記録する (例外時も). `_phase` lifecycle を
    触らないため、ネスト内の `start()`/`end()` ペアと安全に共存できる.
    """

    __slots__ = ("_reporter", "_name", "_t0")

    def __init__(self, reporter: ProgressReporter, name: str) -> None:
        self._reporter = reporter
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_PhaseTimerCM":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._t0
        self._reporter._record_duration(self._name, elapsed)
        # 例外は握らない (None を返して伝播).


__all__ = ["ProgressReporter"]
