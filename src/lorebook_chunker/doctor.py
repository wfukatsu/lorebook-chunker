"""`doctor` subcommand — 環境検証 (Python / 依存 / モデル / API キー / 書き込み可否).

U7 で追加. `operator` が「この環境で本当に走るか」を数秒で確認できる手段を
提供する. チェック結果は以下の 3 段階:

- **pass**   — 問題なし
- **warning**— 動作自体は可能だが sub-optimal (e.g. `charset-normalizer` 未導入)
- **fail**   — critical (Python バージョン、`spacy` / `ginza` / モデル未導入)

## Exit code 空間は ingest 系と独立

本モジュールは ``LorebookError`` 階層 (exit 2-17) と **別空間** の exit code
(0 / 1 / 18) を採用する:

- ``0``  全 check が pass
- ``1``  warning のみ (fail ゼロ)
- ``18`` fail が 1 件以上 (env-critical)

理由: ``EncodingError=12`` や ``AtomicSwapError=16`` と doctor の critical
failure を同一コードに載せると operator がコード分岐不能になる. 18 を確保して
意味的独立を保つ. そのため本モジュールは ``DoctorCritical`` のような
``LorebookError`` サブクラスを **定義しない** — doctor は独立した subcommand
surface として exit code を返す.

``ingest --dry-run`` が doctor を内部的に呼び出すときは、doctor の失敗時に
その exit 18 をそのまま propagate する (ingest 側が 12/16 に remap しない).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


# ---- Exit codes (独立空間) ------------------------------------------------

EXIT_OK = 0
EXIT_WARNING = 1
EXIT_CRITICAL = 18

# ---- Status constants ----------------------------------------------------

STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_FAIL = "fail"


# ---- Supported ranges ----------------------------------------------------

# pyproject.toml と一致させる: Python 3.11/3.12 のみ. 3.13 は tokenizers<0.14
# の wheel が未配布で Ginza 5.2 スタックが動かない.
_PYTHON_MIN = (3, 11)
_PYTHON_MAX = (3, 13)  # exclusive

# spacy>=3.7,<3.9 (Ginza 5.2 compound_splitter factory が 3.9 config 検証で
# 弾かれる既知の互換性境界).
_SPACY_MIN = (3, 7)
_SPACY_MAX = (3, 9)  # exclusive

# ginza>=5.2
_GINZA_MIN = (5, 2)


# ---- Check result shape --------------------------------------------------


@dataclass
class CheckResult:
    """1 つの check の結果. text / JSON 出力共通フォーマット."""

    name: str
    status: str  # "pass" | "warning" | "fail"
    detail: str | None = None
    hint: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
        }


@dataclass
class DoctorConfig:
    """doctor CLI args を dataclass に集約."""

    backend: str | None = None           # "anthropic" | "ollama" | None
    check_bench: bool = False             # `--check-bench`: ranx probe
    output_dir: Path | None = None         # 書き込み可否を検査する候補
    json_output: bool = False
    quiet: bool = False
    ollama_model: str | None = None        # 既定モデルは backend client の default
    ollama_timeout_seconds: float = 5.0


@dataclass
class DoctorSummary:
    """全 check の集計 + exit code."""

    passed: int = 0
    warnings: int = 0
    failures: int = 0
    checks: list[CheckResult] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "failures": self.failures,
            "checks": [c.to_jsonable() for c in self.checks],
        }

    @property
    def exit_code(self) -> int:
        if self.failures > 0:
            return EXIT_CRITICAL
        if self.warnings > 0:
            return EXIT_WARNING
        return EXIT_OK


# ---- Individual check functions ------------------------------------------


def check_python_version() -> CheckResult:
    """Python ``sys.version_info`` が ``>=3.11,<3.13`` かを確認."""
    vi = sys.version_info
    current = f"{vi.major}.{vi.minor}.{vi.micro}"
    if (vi.major, vi.minor) < _PYTHON_MIN or (vi.major, vi.minor) >= _PYTHON_MAX:
        return CheckResult(
            name="python_version",
            status=STATUS_FAIL,
            detail=f"Python {current} (required: >=3.11,<3.13)",
            hint=(
                "install Python 3.11 or 3.12; 3.13+ は tokenizers<0.14 の wheel "
                "が未配布で Ginza 5.2 が動作しない"
            ),
        )
    return CheckResult(
        name="python_version",
        status=STATUS_PASS,
        detail=f"Python {current}",
    )


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """"3.8.14" → (3, 8, 14). 数字以外の suffix (e.g. "rc1") は無視."""
    parts: list[int] = []
    for p in version_str.split("."):
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        parts.append(int(num))
    return tuple(parts)


def check_spacy() -> CheckResult:
    """``spacy`` が import 可能かつ ``>=3.7,<3.9``."""
    try:
        import spacy  # type: ignore
    except ImportError as e:
        return CheckResult(
            name="spacy",
            status=STATUS_FAIL,
            detail=f"spacy not importable: {e}",
            hint="pip install 'spacy>=3.7,<3.9'",
        )
    ver = getattr(spacy, "__version__", "0.0.0")
    parts = _parse_version_tuple(ver)
    mm = parts[:2] if len(parts) >= 2 else (0, 0)
    if mm < _SPACY_MIN or mm >= _SPACY_MAX:
        return CheckResult(
            name="spacy",
            status=STATUS_FAIL,
            detail=f"spacy {ver} (required: >=3.7,<3.9)",
            hint="pip install 'spacy>=3.7,<3.9'",
        )
    return CheckResult(name="spacy", status=STATUS_PASS, detail=f"spacy {ver}")


def check_ginza() -> CheckResult:
    """``ginza`` が import 可能かつ ``>=5.2``."""
    try:
        import ginza  # type: ignore
    except ImportError as e:
        return CheckResult(
            name="ginza",
            status=STATUS_FAIL,
            detail=f"ginza not importable: {e}",
            hint="pip install 'ginza>=5.2'",
        )
    # ginza は __version__ を持たないことがあるので importlib.metadata にフォールバック.
    ver = getattr(ginza, "__version__", None)
    if ver is None:
        try:
            from importlib.metadata import version as _metadata_version

            ver = _metadata_version("ginza")
        except Exception:  # pragma: no cover
            ver = "0.0.0"
    parts = _parse_version_tuple(str(ver))
    mm = parts[:2] if len(parts) >= 2 else (0, 0)
    if mm < _GINZA_MIN:
        return CheckResult(
            name="ginza",
            status=STATUS_FAIL,
            detail=f"ginza {ver} (required: >=5.2)",
            hint="pip install 'ginza>=5.2'",
        )
    return CheckResult(name="ginza", status=STATUS_PASS, detail=f"ginza {ver}")


def _spacy_load_ginza_with_shim(model_name: str) -> Any:
    """``analyzer.JapaneseAnalyzer.__init__`` と同じ Ginza 5.2 × spacy 3.8 互換
    shim を適用して ``spacy.load`` する.

    compound_splitter の ``split_mode`` を明示的に渡さないと、spacy 3.8 の厳格
    な Config 検証で ``None is not <class 'str'>`` エラーになる.
    """
    import spacy  # type: ignore

    # analyzer.DEFAULT_SPLIT_MODE と同一の既定値. 循環 import を避けるため本モジュール内に
    # ハードコードする (analyzer.py 自体が重い import を持つため).
    split_mode = "C"
    return spacy.load(
        model_name,
        exclude=["bunsetu_recognizer"],
        config={"components": {"compound_splitter": {"split_mode": split_mode}}},
    )


def check_ja_ginza_model() -> CheckResult:
    """``ja_ginza_electra`` (既定) または ``ja_ginza`` (fast) が ``spacy.load`` 可能.

    Ginza 5.2 × spacy 3.8 互換シム (compound_splitter split_mode="C" 明示) を
    適用してから load する — analyzer.JapaneseAnalyzer と同じ契約.

    ELECTRA 版だけ入っていれば pass. ``ja_ginza_electra`` が無く ``ja_ginza``
    (fast) のみの場合は warning (fast mode だが ingest は動く).
    どちらも無ければ fail.
    """
    try:
        import spacy  # type: ignore  # noqa: F401
    except ImportError:
        return CheckResult(
            name="ja_ginza_model",
            status=STATUS_FAIL,
            detail="spacy not importable",
            hint="pip install 'spacy>=3.7,<3.9'",
        )
    # ELECTRA (既定) を shim 付きで試行.
    try:
        _spacy_load_ginza_with_shim("ja_ginza_electra")
        return CheckResult(
            name="ja_ginza_model",
            status=STATUS_PASS,
            detail="ja_ginza_electra loaded",
        )
    except Exception as e_electra:
        # fallback: fast モデルも shim 付きで試行.
        try:
            _spacy_load_ginza_with_shim("ja_ginza")
            return CheckResult(
                name="ja_ginza_model",
                status=STATUS_WARNING,
                detail=(
                    "ja_ginza_electra unavailable, but ja_ginza (fast) loadable — "
                    "run ingest with --analyzer-backend ginza"
                ),
                hint=(
                    "for full NER accuracy install ja_ginza_electra "
                    "(`pip install -e '.'`)"
                ),
            )
        except Exception as e_fast:
            return CheckResult(
                name="ja_ginza_model",
                status=STATUS_FAIL,
                detail=(
                    f"neither ja_ginza_electra nor ja_ginza loadable "
                    f"(electra={type(e_electra).__name__}, ja_ginza={type(e_fast).__name__})"
                ),
                hint=(
                    "pip install -e '.' (installs ja_ginza_electra wheel) "
                    "or pip install -e '.[fast]' (ja_ginza only)"
                ),
            )


def check_charset_normalizer() -> CheckResult:
    """``charset-normalizer`` 未導入は warning のみ (`[full]` extra でオプトイン)."""
    try:
        importlib.import_module("charset_normalizer")
    except ImportError:
        return CheckResult(
            name="charset_normalizer",
            status=STATUS_WARNING,
            detail=(
                "charset-normalizer not installed — --encoding auto は "
                "UTF-8 (BOM 有無) のみ通過、非 UTF-8 入力は skip されます"
            ),
            hint=(
                "pip install -e '.[full]' で Shift-JIS / CP932 / EUC-JP 等を "
                "auto 検出できます"
            ),
        )
    return CheckResult(
        name="charset_normalizer",
        status=STATUS_PASS,
        detail="charset-normalizer importable",
    )


def check_anthropic(*, required: bool) -> CheckResult:
    """``anthropic`` client が import 可能、``required=True`` なら key も検査.

    - client 未導入: ``required=True`` なら fail、そうでなければ warning (skip)
    - key 未設定 で ``required=True``: warning (config ミス、ingest 時に exit 3)
    """
    try:
        importlib.import_module("anthropic")
    except ImportError:
        if required:
            return CheckResult(
                name="anthropic",
                status=STATUS_FAIL,
                detail="anthropic not importable (backend=anthropic requested)",
                hint="pip install 'anthropic>=0.34'",
            )
        return CheckResult(
            name="anthropic",
            status=STATUS_WARNING,
            detail="anthropic not importable (no backend requested)",
            hint="pip install 'anthropic>=0.34' を `--llm-backend anthropic` 利用時に必要",
        )
    if required:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return CheckResult(
                name="anthropic",
                status=STATUS_FAIL,
                detail="ANTHROPIC_API_KEY not set (backend=anthropic requested)",
                hint="export ANTHROPIC_API_KEY=sk-...",
            )
    return CheckResult(
        name="anthropic",
        status=STATUS_PASS,
        detail="anthropic client importable"
        + (" (+ ANTHROPIC_API_KEY set)" if required else ""),
    )


def _ollama_show_with_timeout(
    model: str, timeout_seconds: float
) -> tuple[bool, str | None]:
    """``ollama.show(model)`` を timeout 付きで叩く.

    ``concurrent.futures.ThreadPoolExecutor`` を使う理由:
      - ``signal.alarm`` は main thread でしか使えず, pytest ランナー内で
        installed signal handler と衝突する.
      - ``socket`` レベルの timeout は HTTP client のみに効き、gRPC 等に
        拡張された場合 portability を失う.
    ThreadPoolExecutor ``.result(timeout=...)`` は worker thread を dangling
    させる欠点はあるが、doctor は short-lived process なのでリークしても
    プロセス終了で回収される.

    戻り値: ``(ok, error_message)``.
    """
    import concurrent.futures

    try:
        ollama = importlib.import_module("ollama")
    except ImportError as e:
        return False, f"ollama not importable: {e}"

    def _call() -> Any:
        return ollama.show(model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_call)
        try:
            fut.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return False, f"ollama.show({model!r}) timed out after {timeout_seconds:.1f}s"
        except Exception as e:
            return False, f"ollama.show({model!r}) failed: {type(e).__name__}: {e}"
    return True, None


def check_ollama(
    *,
    required: bool,
    model: str | None = None,
    timeout_seconds: float = 5.0,
) -> CheckResult:
    """``ollama`` client が import 可能、``required=True`` なら ``ollama.show(model)`` も確認."""
    try:
        importlib.import_module("ollama")
    except ImportError:
        if required:
            return CheckResult(
                name="ollama",
                status=STATUS_FAIL,
                detail="ollama not importable (backend=ollama requested)",
                hint="pip install 'ollama>=0.3'",
            )
        return CheckResult(
            name="ollama",
            status=STATUS_WARNING,
            detail="ollama not importable (no backend requested)",
            hint="pip install 'ollama>=0.3' を `--llm-backend ollama` 利用時に必要",
        )
    if required:
        # 既定モデル: CLI の OllamaLLMClient default.
        target_model = model or "qwen2.5:7b-instruct-q4_K_M"
        ok, err = _ollama_show_with_timeout(target_model, timeout_seconds)
        if not ok:
            # reachable 判定: TimeoutError または ConnectionError 系は fail、
            # 単にモデル未 pull は warning. エラーメッセージから推定する.
            err_msg = err or ""
            lower = err_msg.lower()
            if (
                "timed out" in lower
                or "connection" in lower
                or "refused" in lower
                or "unreachable" in lower
            ):
                return CheckResult(
                    name="ollama",
                    status=STATUS_FAIL,
                    detail=err_msg,
                    hint=(
                        "ollama serve が起動しているか、"
                        "`ollama ps` / `curl http://localhost:11434` で確認"
                    ),
                )
            # モデル未 pull / 不明モデル等.
            return CheckResult(
                name="ollama",
                status=STATUS_WARNING,
                detail=err_msg,
                hint=f"ollama pull {target_model}",
            )
        return CheckResult(
            name="ollama",
            status=STATUS_PASS,
            detail=f"ollama.show({target_model!r}) OK",
        )
    return CheckResult(
        name="ollama",
        status=STATUS_PASS,
        detail="ollama client importable",
    )


def check_ranx() -> CheckResult:
    """``ranx`` が import 可能か. 未導入は warning ([bench] extra)."""
    try:
        importlib.import_module("ranx")
    except ImportError:
        return CheckResult(
            name="ranx",
            status=STATUS_WARNING,
            detail="ranx not installed ([bench] extra 未導入)",
            hint="pip install -e '.[bench]' で RAG 評価 (scripts/bench.py) が動く",
        )
    return CheckResult(name="ranx", status=STATUS_PASS, detail="ranx importable")


def check_output_dir_writable(output_dir: Path) -> CheckResult:
    """``output_dir`` の親ディレクトリに書き込み可能か確認.

    親が存在しない → fail ではなく warning (ingest 側で mkdir で作る想定).
    tempfile でテスト書き込み → 成功すれば pass.
    """
    parent = output_dir.parent
    if not parent.exists():
        return CheckResult(
            name="output_dir_writable",
            status=STATUS_WARNING,
            detail=f"parent {parent} does not exist (ingest will mkdir)",
            hint=f"mkdir -p {parent}",
        )
    if not parent.is_dir():
        return CheckResult(
            name="output_dir_writable",
            status=STATUS_FAIL,
            detail=f"parent {parent} exists but is not a directory",
            hint=None,
        )
    # 実際の書き込みテスト.
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(parent), prefix=".lorebook-chunker-doctor-", delete=True
        ) as _:
            pass
    except OSError as e:
        return CheckResult(
            name="output_dir_writable",
            status=STATUS_FAIL,
            detail=f"cannot create temp file in {parent}: {e}",
            hint="check filesystem permissions / disk space",
        )
    return CheckResult(
        name="output_dir_writable",
        status=STATUS_PASS,
        detail=f"parent {parent} writable",
    )


# ---- Composition ---------------------------------------------------------


def run_checks(cfg: DoctorConfig) -> DoctorSummary:
    """cfg に応じて適切なチェックを実行し、``DoctorSummary`` を返す."""
    summary = DoctorSummary()

    def _record(result: CheckResult) -> None:
        summary.checks.append(result)
        if result.status == STATUS_PASS:
            summary.passed += 1
        elif result.status == STATUS_WARNING:
            summary.warnings += 1
        elif result.status == STATUS_FAIL:
            summary.failures += 1

    _record(check_python_version())
    _record(check_spacy())
    _record(check_ginza())
    _record(check_ja_ginza_model())
    _record(check_charset_normalizer())

    # LLM backend probes: backend= が指定されたら required フラグを立てて深く見る.
    _record(check_anthropic(required=(cfg.backend == "anthropic")))
    _record(
        check_ollama(
            required=(cfg.backend == "ollama"),
            model=cfg.ollama_model,
            timeout_seconds=cfg.ollama_timeout_seconds,
        )
    )

    if cfg.check_bench:
        _record(check_ranx())

    if cfg.output_dir is not None:
        _record(check_output_dir_writable(cfg.output_dir))

    return summary


# ---- Rendering -----------------------------------------------------------


_STATUS_MARKS = {
    STATUS_PASS: "PASS",
    STATUS_WARNING: "WARN",
    STATUS_FAIL: "FAIL",
}


def render_summary_text(summary: DoctorSummary) -> str:
    """人間可読出力. 各 check を 1 行で表示し、最後に集計を付ける."""
    lines: list[str] = []
    for c in summary.checks:
        mark = _STATUS_MARKS.get(c.status, c.status.upper())
        line = f"[{mark}] {c.name}"
        if c.detail:
            line += f" — {c.detail}"
        lines.append(line)
        if c.hint and c.status != STATUS_PASS:
            lines.append(f"       hint: {c.hint}")
    lines.append("")
    lines.append(
        f"summary: {summary.passed} passed, "
        f"{summary.warnings} warnings, {summary.failures} failures "
        f"(exit {summary.exit_code})"
    )
    return "\n".join(lines)


# ---- CLI entry -----------------------------------------------------------

# 日本語 epilog は doctor 固有の exit code 空間 (0/1/18) を明示.
DOCTOR_EPILOG = """\
exit codes (ingest の 2-17 とは独立した namespace):
  0   全 check が pass
  1   warnings のみ (failures ゼロ)
  18  1 件以上の env-critical failure (Python/spacy/ginza/model 等)
"""


def add_doctor_subparser(subparsers: argparse._SubParsersAction) -> None:
    """``lorebook-chunker doctor`` subparser を登録する."""
    from lorebook_chunker.cli import IDENTITY_BANNER

    p = subparsers.add_parser(
        "doctor",
        help="環境検証 (Python / 依存 / モデル / API キー / 書き込み可否)",
        description=IDENTITY_BANNER,
        epilog=DOCTOR_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--backend",
        choices=["anthropic", "ollama"],
        default=None,
        help=(
            "LLM バックエンドに応じた追加 check を有効化. "
            "anthropic 指定時は ANTHROPIC_API_KEY の存在を検査、"
            "ollama 指定時は `ollama.show(<model>)` で疎通を確認 (既定 5s タイムアウト)."
        ),
    )
    p.add_argument(
        "--ollama-model",
        default=None,
        metavar="MODEL",
        help="`--backend ollama` 時に probe するモデル名 (既定: qwen2.5:7b-instruct-q4_K_M)",
    )
    p.add_argument(
        "--check-bench",
        action="store_true",
        help="ranx ([bench] extra) の probe を追加. 既定では bench 依存は検査しない.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help=(
            "出力候補ディレクトリ. 親ディレクトリへの書き込み可否 "
            "(tempfile 経由) を検査する. 未指定時はこの check をスキップ."
        ),
    )
    p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="構造化 JSON ({checks, summary}) を stdout に出力 (機械可読).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="identity banner を抑止.",
    )


def _build_cfg_from_args(args: argparse.Namespace) -> DoctorConfig:
    from lorebook_chunker.errors import ConfigError

    backend = getattr(args, "backend", None)
    out_dir_arg = getattr(args, "output_dir", None)
    out_dir = Path(out_dir_arg) if out_dir_arg else None
    cfg = DoctorConfig(
        backend=backend,
        check_bench=getattr(args, "check_bench", False),
        output_dir=out_dir,
        json_output=getattr(args, "json_output", False),
        quiet=getattr(args, "quiet", False),
        ollama_model=getattr(args, "ollama_model", None),
    )
    # Sanity check: backend=ollama で ollama_model 指定なし → OK (default 使用).
    # backend=anthropic で ollama_model 指定は意味を持たないが無害なので許容.
    return cfg


def run_doctor(args: argparse.Namespace) -> int:
    """``doctor`` サブコマンド本体. stdout/stderr に出力して exit code を返す."""
    from lorebook_chunker.cli import IDENTITY_BANNER

    cfg = _build_cfg_from_args(args)
    if not cfg.quiet and not cfg.json_output:
        print(IDENTITY_BANNER, file=sys.stderr)
    summary = run_checks(cfg)
    if cfg.json_output:
        print(json.dumps(summary.to_jsonable(), ensure_ascii=False))
    else:
        print(render_summary_text(summary))
    return summary.exit_code


__all__ = [
    "EXIT_OK",
    "EXIT_WARNING",
    "EXIT_CRITICAL",
    "STATUS_PASS",
    "STATUS_WARNING",
    "STATUS_FAIL",
    "CheckResult",
    "DoctorConfig",
    "DoctorSummary",
    "DOCTOR_EPILOG",
    "check_python_version",
    "check_spacy",
    "check_ginza",
    "check_ja_ginza_model",
    "check_charset_normalizer",
    "check_anthropic",
    "check_ollama",
    "check_ranx",
    "check_output_dir_writable",
    "run_checks",
    "render_summary_text",
    "run_doctor",
    "add_doctor_subparser",
]
