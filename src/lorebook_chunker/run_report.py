"""U5: `run_report.json` v1 writer + schema dataclass.

各 ingest run の成功/失敗をまとめた canonical summary を `output_dir/run_report.json`
に atomic write する. U5 の pre-release まで存在した F-036 ad-hoc summary writer の
置換であり、schema v1 の key set を凍結する.

## Schema stability (v1)

以下の top-level keys は v1 で **安定** する. 将来の変更は additive-only
(既存 key の削除/リネーム/値域縮小は `schema_version` bump を伴う):

- `schema_version` (int, 1 固定)
- `lorebook_chunker_version` (str)
- `exit_code` (int)
- `exit_reason` (dict | None): `{"class", "message", "context"}`
- `started_at` / `completed_at` (ISO-8601 str, timezone offset 付き)
- `duration_seconds` (float)
- `phase_durations_seconds` (dict): 6 固定 key
  `analyzer_init` / `chunking` / `tfidf` / `ner` / `wiki` / `swap`
- `input` (dict): `input_dir` / `recursive` / `globs` / `encoding_option` /
  `files_processed` / `files_skipped`
- `output` (dict): `output_dir` / `chunks_generated` / `entities_generated` /
  `wiki_pages_written`
- `analyzer` (dict): `model_name` / `model_version` / `model_sha256` /
  `spacy_version` / `ginza_version` / `sudachi_dict`
- `llm` (dict): `backend` / `model_id` / `total_input_tokens` /
  `total_output_tokens`
- `warnings` (list[str])

## Atomic write

`wiki.py` の `ManifestStore.save` と同じ pattern:
same-filesystem `tempfile.mkstemp` → fsync → `os.replace` → parent dir fsync.
失敗時は `RunReportError(exit_code=17)` を raise する.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lorebook_chunker import __version__ as _LOREBOOK_CHUNKER_VERSION
from lorebook_chunker.errors import RunReportError

# U5: phase_durations_seconds の stable key set (v1 固定).
PHASE_NAMES: tuple[str, ...] = (
    "analyzer_init",
    "chunking",
    "tfidf",
    "ner",
    "wiki",
    "swap",
)


@dataclass
class RunReport:
    """run_report.json の機械可読表現. `to_json_dict` で JSON serializable dict を返す."""

    SCHEMA_VERSION: int = 1  # class 定数相当. `to_json_dict` で参照.

    lorebook_chunker_version: str = _LOREBOOK_CHUNKER_VERSION
    exit_code: int = 0
    exit_reason: dict[str, Any] | None = None
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    phase_durations_seconds: dict[str, float] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    analyzer: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON serializable dict. phase key set を 6 phase に padding する."""
        phases = {name: 0.0 for name in PHASE_NAMES}
        for name, dur in self.phase_durations_seconds.items():
            if name in phases:
                phases[name] = float(dur)
        # schema_version は class 定数由来 (runtime mutation を受けない).
        return {
            "schema_version": self.SCHEMA_VERSION,
            "lorebook_chunker_version": self.lorebook_chunker_version,
            "exit_code": int(self.exit_code),
            "exit_reason": self.exit_reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": float(self.duration_seconds),
            "phase_durations_seconds": phases,
            "input": dict(self.input),
            "output": dict(self.output),
            "analyzer": dict(self.analyzer),
            "llm": dict(self.llm),
            "warnings": list(self.warnings),
        }


def write(path: Path, report: RunReport) -> None:
    """`run_report.json` を atomic write する.

    失敗時は `RunReportError(exit_code=17)` を raise. caller (CLI boundary)
    が `result.errors` に記録し exit code 17 で終了する.

    atomic write pattern は `wiki.py:223-260` と `analyzer.py:629-646` を踏襲:
    same-fs tempfile → fsync → os.replace → parent-dir fsync.
    """
    path = Path(path)
    tmp_path: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".run_report.", suffix=".tmp"
        )
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(
                report.to_json_dict(),
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = None  # successfully replaced, no cleanup needed
        # 親ディレクトリの dirent も同期 (POSIX only)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, NotImplementedError):  # Windows など
            pass
    except OSError as e:
        raise RunReportError(
            f"run_report.json write failed: {e}",
            path=str(path),
            reason=type(e).__name__,
            errno=getattr(e, "errno", None) or 0,
        ) from e
    except Exception as e:
        # tempfile.mkstemp mocked to raise (test scenario) 等を包括.
        raise RunReportError(
            f"run_report.json write failed: {e}",
            path=str(path),
            reason=type(e).__name__,
        ) from e
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - best effort
                pass


__all__ = ["RunReport", "PHASE_NAMES", "write"]
