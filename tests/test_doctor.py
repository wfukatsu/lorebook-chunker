"""U7: doctor subcommand + ingest --dry-run tests.

Coverage (plan U7 Test scenarios):

- Happy path: clean env doctor → exit 0
- Happy path: doctor --json が構造化 JSON を emit
- Happy path: doctor --backend=anthropic + ANTHROPIC_API_KEY 未設定 → exit 18
- Happy path: charset-normalizer 未導入 simulation → exit 1 (warning)
- Edge case: ingest --dry-run → exit 0、out/ 未作成、stdout JSON
- Edge case: ingest --dry-run with 0 files → exit 2 (ConfigError)
- Edge case: --dry-run が encoding probe を実施
- Edge case: doctor exit code (0/1/18) が ingest 系 (2-17) と衝突しない
- Error path: Python 3.10 simulation → doctor exit 18
- Error path: doctor invalid-arg → argparse exit 2
- Integration: README に 5 profile が記載
- Integration: subprocess 経由で CLI を呼ぶ (real CLI contract)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lorebook_chunker import doctor as doctor_mod
from lorebook_chunker.doctor import (
    EXIT_CRITICAL,
    EXIT_OK,
    EXIT_WARNING,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    CheckResult,
    DoctorConfig,
    DoctorSummary,
    check_anthropic,
    check_charset_normalizer,
    check_ginza,
    check_ja_ginza_model,
    check_ollama,
    check_output_dir_writable,
    check_python_version,
    check_ranx,
    check_spacy,
    render_summary_text,
    run_checks,
)


# ---- Unit tests for individual checks -----------------------------------


def test_check_python_version_pass_on_311_312() -> None:
    """現在の Python が 3.11 or 3.12 なら pass を返す (CI 環境を前提)."""
    r = check_python_version()
    # 現状の .venv311 は 3.11 で動いている想定.
    vi = sys.version_info
    if (vi.major, vi.minor) in ((3, 11), (3, 12)):
        assert r.status == STATUS_PASS
    else:  # pragma: no cover - future version audit
        assert r.status == STATUS_FAIL


def test_check_python_version_fail_on_310(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python 3.10 simulation — sys.version_info を差し替えて fail path を確認."""
    from collections import namedtuple

    VI = namedtuple("VI", ["major", "minor", "micro", "releaselevel", "serial"])
    fake = VI(3, 10, 14, "final", 0)
    monkeypatch.setattr(sys, "version_info", fake)
    r = check_python_version()
    assert r.status == STATUS_FAIL
    assert "3.10" in (r.detail or "")
    assert "3.11" in (r.hint or "") or "3.12" in (r.hint or "")


def test_check_python_version_fail_on_313(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python 3.13 simulation — 上限 exclusive なので fail."""
    from collections import namedtuple

    VI = namedtuple("VI", ["major", "minor", "micro", "releaselevel", "serial"])
    fake = VI(3, 13, 0, "final", 0)
    monkeypatch.setattr(sys, "version_info", fake)
    r = check_python_version()
    assert r.status == STATUS_FAIL


def test_check_spacy_pass() -> None:
    """本プロジェクトは spacy >=3.7,<3.9 を依存にするため pass するはず."""
    r = check_spacy()
    assert r.status == STATUS_PASS
    assert "spacy" in (r.detail or "").lower()


def test_check_ginza_pass() -> None:
    """ginza >=5.2 が入っている前提."""
    r = check_ginza()
    assert r.status == STATUS_PASS


def test_check_ja_ginza_model_pass() -> None:
    """ja_ginza_electra が import/load 可能 (wheel 経由で導入済み)."""
    r = check_ja_ginza_model()
    # full install では pass、fast install では warning の可能性あり.
    assert r.status in (STATUS_PASS, STATUS_WARNING)


def test_check_charset_normalizer_pass_when_installed() -> None:
    """[full] で入っている場合は pass. 未導入環境でも検査自体は動く."""
    r = check_charset_normalizer()
    assert r.status in (STATUS_PASS, STATUS_WARNING)


def test_check_charset_normalizer_warning_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """charset-normalizer 未導入シミュレーション → warning (exit 1) + install hint."""
    import importlib

    orig_import = importlib.import_module

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "charset_normalizer":
            raise ImportError("simulated missing charset_normalizer")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    r = check_charset_normalizer()
    assert r.status == STATUS_WARNING
    # install hint が具体的 (`[full]` extra 言及).
    assert "full" in (r.hint or "")


def test_check_anthropic_missing_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """backend=anthropic で ANTHROPIC_API_KEY 未設定 → fail + export hint."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = check_anthropic(required=True)
    assert r.status == STATUS_FAIL
    assert "ANTHROPIC_API_KEY" in (r.detail or "")
    assert "ANTHROPIC_API_KEY" in (r.hint or "")


def test_check_anthropic_not_required_passes_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """key 未設定でも required=False なら pass (client import できていれば)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = check_anthropic(required=False)
    # anthropic SDK は dependencies に入っているので pass のはず.
    assert r.status == STATUS_PASS


def test_check_anthropic_with_key_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """backend=anthropic + key セットで pass."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
    r = check_anthropic(required=True)
    assert r.status == STATUS_PASS


def test_check_ollama_not_required() -> None:
    """required=False なら client import できていれば pass."""
    r = check_ollama(required=False)
    assert r.status == STATUS_PASS


def test_check_ollama_required_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """required=True + connection error → fail with hint to start ollama serve."""
    import lorebook_chunker.doctor as dm

    def _fake_probe(model: str, timeout_seconds: float) -> tuple[bool, str | None]:
        return False, "connection refused (mock)"

    monkeypatch.setattr(dm, "_ollama_show_with_timeout", _fake_probe)
    r = check_ollama(required=True, model="some-model")
    assert r.status == STATUS_FAIL
    assert "ollama serve" in (r.hint or "")


def test_check_ollama_required_model_not_pulled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """required=True + model not pulled → warning (not fail), hint to pull."""
    import lorebook_chunker.doctor as dm

    def _fake_probe(model: str, timeout_seconds: float) -> tuple[bool, str | None]:
        return False, "ollama.show('xxx') failed: ModelNotFound: pull the model"

    monkeypatch.setattr(dm, "_ollama_show_with_timeout", _fake_probe)
    r = check_ollama(required=True, model="missing-model")
    assert r.status == STATUS_WARNING
    assert "pull" in (r.hint or "")


def test_check_ranx_warning_when_missing() -> None:
    """ranx は [bench] extra 用. 通常の dev 環境では未導入 → warning."""
    r = check_ranx()
    # 環境によって変わるが warning か pass.
    assert r.status in (STATUS_WARNING, STATUS_PASS)
    if r.status == STATUS_WARNING:
        assert "bench" in (r.hint or "")


def test_check_output_dir_writable_pass(tmp_path: Path) -> None:
    """親が存在して writable なら pass."""
    r = check_output_dir_writable(tmp_path / "out")
    assert r.status == STATUS_PASS


def test_check_output_dir_writable_nonexistent_parent(tmp_path: Path) -> None:
    """親ディレクトリが無ければ warning (ingest 側で mkdir する想定)."""
    r = check_output_dir_writable(tmp_path / "does_not_exist" / "out")
    assert r.status == STATUS_WARNING


# ---- Summary / exit code mapping ----------------------------------------


def test_doctor_summary_exit_code_logic() -> None:
    """failures > 0 → 18、warnings > 0 → 1、それ以外 → 0."""
    s = DoctorSummary(passed=5, warnings=0, failures=0)
    assert s.exit_code == EXIT_OK

    s = DoctorSummary(passed=4, warnings=1, failures=0)
    assert s.exit_code == EXIT_WARNING

    s = DoctorSummary(passed=3, warnings=0, failures=1)
    assert s.exit_code == EXIT_CRITICAL

    s = DoctorSummary(passed=3, warnings=1, failures=1)
    assert s.exit_code == EXIT_CRITICAL  # failure が優先


def test_run_checks_assembles_summary() -> None:
    """run_checks が clean env で DoctorSummary を組み立てる."""
    summary = run_checks(DoctorConfig())
    # 少なくとも python/spacy/ginza/model/charset/anthropic/ollama の 7 item.
    assert len(summary.checks) >= 7
    total = summary.passed + summary.warnings + summary.failures
    assert total == len(summary.checks)


def test_doctor_exit_codes_independent_from_ingest_namespace() -> None:
    """doctor exit codes (0/1/18) と ingest/LorebookError 系 (2-17) が衝突しない.

    intersection は {0} のみ (doctor success == ingest success).
    """
    from lorebook_chunker import errors as err_mod

    ingest_codes = set(err_mod.EXIT_CODE_MAP.values())
    # ingest 系は 2/3/4/5/6/11..17 (ConfigError..RunReportError).
    doctor_codes = {EXIT_OK, EXIT_WARNING, EXIT_CRITICAL}  # {0, 1, 18}
    intersection = ingest_codes & doctor_codes
    assert intersection == set(), (
        f"doctor / ingest exit code spaces must be disjoint, found overlap: {intersection}"
    )
    # 18 は ingest 系に含まれていない.
    assert 18 not in ingest_codes
    # 1 も含まれていない.
    assert 1 not in ingest_codes


# ---- CLI integration (via subprocess, preserves real CLI contract) -----


def _run_cli(
    *argv: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """subprocess で lorebook-chunker CLI を呼ぶ helper."""
    cmd = [sys.executable, "-m", "lorebook_chunker", *argv]
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=merged_env,
        cwd=str(cwd) if cwd else None,
    )


def test_cli_doctor_clean_env_exits_0_or_1() -> None:
    """CLI 経由で doctor 実行 → clean env では 0 (pass) もしくは 1 (charset 等の warning)."""
    res = _run_cli("doctor")
    assert res.returncode in (EXIT_OK, EXIT_WARNING), (
        f"unexpected exit {res.returncode}, stdout={res.stdout!r}, stderr={res.stderr!r}"
    )
    # 出力に PASS / WARN マーカーが少なくとも 1 つ登場する.
    assert "PASS" in res.stdout or "WARN" in res.stdout


def test_cli_doctor_json_emits_structured_payload() -> None:
    """--json で {checks: [...], passed/warnings/failures: int} を emit."""
    res = _run_cli("doctor", "--json")
    assert res.returncode in (EXIT_OK, EXIT_WARNING, EXIT_CRITICAL)
    data = json.loads(res.stdout)
    assert "checks" in data and isinstance(data["checks"], list)
    assert "passed" in data and "warnings" in data and "failures" in data
    # 各 check は name/status/detail/hint を持つ.
    for c in data["checks"]:
        assert "name" in c and "status" in c
        assert c["status"] in ("pass", "warning", "fail")


def test_cli_doctor_backend_anthropic_no_key_exits_18(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """backend=anthropic + ANTHROPIC_API_KEY 未設定 → exit 18 + hint."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    # subprocess 側の env を明示的に制御.
    cmd = [
        sys.executable,
        "-m",
        "lorebook_chunker",
        "doctor",
        "--backend",
        "anthropic",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == EXIT_CRITICAL
    data = json.loads(res.stdout)
    anthropic_check = next(
        (c for c in data["checks"] if c["name"] == "anthropic"), None
    )
    assert anthropic_check is not None
    assert anthropic_check["status"] == "fail"
    assert "ANTHROPIC_API_KEY" in (anthropic_check["hint"] or "")


def test_cli_doctor_invalid_arg_exits_2() -> None:
    """doctor に未知のサブ引数 → argparse が exit 2."""
    res = _run_cli("doctor", "--no-such-flag")
    assert res.returncode == 2  # argparse default error code


def test_cli_help_lists_doctor() -> None:
    """--help の出力に doctor が subcommand として表示される."""
    res = _run_cli("--help")
    assert res.returncode == 0
    # subcommand 名が usage 行 or choices に出る.
    assert "doctor" in res.stdout
    # 既存も生きている.
    assert "ingest" in res.stdout
    assert "query" in res.stdout
    assert "lint" in res.stdout


def test_cli_doctor_help_mentions_exit_codes() -> None:
    """doctor --help の epilog に 0/1/18 と ingest namespace との区別が書いてある."""
    res = _run_cli("doctor", "--help")
    assert res.returncode == 0
    assert "0" in res.stdout
    assert "1" in res.stdout
    assert "18" in res.stdout


# ---- ingest --dry-run -------------------------------------------------


def _write_samples(dir_: Path, contents: dict[str, bytes | str]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    for name, body in contents.items():
        target = dir_ / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            target.write_bytes(body)
        else:
            target.write_text(body, encoding="utf-8")
    return dir_


def test_cli_ingest_dry_run_happy_path(tmp_path: Path) -> None:
    """valid input + valid output path → exit 0、out/ 未作成、stdout JSON."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _write_samples(
        input_dir,
        {
            "a.txt": "スカラー商事は本日、重要な発表を行いました。" * 20,
            "b.txt": "田中氏は東京本社で働いています。" * 20,
        },
    )
    res = _run_cli(
        "ingest",
        "--dry-run",
        str(input_dir),
        str(output_dir),
        "--skip-wiki",
        "--quiet",
    )
    assert res.returncode == 0, (
        f"unexpected exit {res.returncode}; stderr={res.stderr}; stdout={res.stdout}"
    )
    # out/ は作られていない.
    assert not output_dir.exists(), "--dry-run must not create output_dir"
    data = json.loads(res.stdout.strip().splitlines()[-1])
    assert data["dry_run"] is True
    assert data["files_discovered"] == 2
    assert data["output_dir_created"] is False
    assert "first_file_encoding_probe" in data
    assert data["first_file_encoding_probe"]["encoding"] in ("utf-8-sig", "utf-8")


def test_cli_ingest_dry_run_zero_files_exits_2(tmp_path: Path) -> None:
    """入力 0 件 → exit 2 (ConfigError no_input_files、ingest 空間)."""
    input_dir = tmp_path / "empty_in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    res = _run_cli(
        "ingest",
        "--dry-run",
        str(input_dir),
        str(output_dir),
        "--skip-wiki",
        "--quiet",
    )
    assert res.returncode == 2
    assert not output_dir.exists()
    data = json.loads(res.stdout.strip().splitlines()[-1])
    assert data["dry_run"] is True
    assert "error" in data
    assert data["error"]["class"] == "ConfigError"
    assert data["error"]["context"].get("reason") == "no_input_files"


def test_cli_ingest_dry_run_encoding_probe_detects_utf8(tmp_path: Path) -> None:
    """先頭ファイルの encoding probe が実行される (detected encoding が summary に載る)."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    # UTF-8 BOM + 日本語.
    _write_samples(
        input_dir,
        {
            "a.txt": (
                "﻿" + "テスト文書です。これは UTF-8-SIG でエンコードされています。" * 10
            ),
        },
    )
    res = _run_cli(
        "ingest",
        "--dry-run",
        str(input_dir),
        str(output_dir),
        "--skip-wiki",
        "--quiet",
    )
    assert res.returncode == 0
    data = json.loads(res.stdout.strip().splitlines()[-1])
    enc = data["first_file_encoding_probe"]["encoding"]
    assert enc == "utf-8-sig"
    # sample_bytes_examined > 0.
    assert data["first_file_encoding_probe"]["sample_bytes_examined"] > 0


def test_cli_ingest_dry_run_does_not_emit_run_report(tmp_path: Path) -> None:
    """dry-run は run_report.json を書かない (output_dir 自体を作らないのが契約)."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    _write_samples(input_dir, {"a.txt": "ダミー文書です。" * 30})
    res = _run_cli(
        "ingest",
        "--dry-run",
        str(input_dir),
        str(output_dir),
        "--skip-wiki",
        "--quiet",
    )
    assert res.returncode == 0
    assert not output_dir.exists()
    assert not (output_dir / "run_report.json").exists()


# ---- README profile integration test ---------------------------------


def test_readme_lists_5_install_profiles() -> None:
    """README に minimal / fast / full / bench / dev の 5 プロファイルが載っている."""
    readme = Path(__file__).resolve().parent.parent / "README.md"
    content = readme.read_text(encoding="utf-8")
    for profile in ("minimal", "fast", "full", "bench", "dev"):
        assert profile in content, f"README missing profile: {profile}"
    # インストール列も全部載っている.
    assert "pip install lorebook-chunker[fast]" in content
    assert "pip install lorebook-chunker[full]" in content
    assert "pip install lorebook-chunker[bench]" in content
    assert "pip install lorebook-chunker[dev]" in content


def test_readme_mentions_doctor_and_dry_run() -> None:
    """README に doctor と --dry-run の使用例と exit code 0/1/18 が書いてある."""
    readme = Path(__file__).resolve().parent.parent / "README.md"
    content = readme.read_text(encoding="utf-8")
    assert "lorebook-chunker doctor" in content
    assert "--dry-run" in content
    # exit code: doctor (0/1/18).
    assert "18" in content


# ---- Render sanity ----------------------------------------------------


def test_render_summary_text_contains_all_checks() -> None:
    """render_summary_text が全 check を 1 行で出し、summary 行で締める."""
    results = [
        CheckResult(name="a", status=STATUS_PASS, detail="ok"),
        CheckResult(
            name="b",
            status=STATUS_WARNING,
            detail="sub-optimal",
            hint="do this",
        ),
        CheckResult(
            name="c", status=STATUS_FAIL, detail="broken", hint="fix that"
        ),
    ]
    s = DoctorSummary(passed=1, warnings=1, failures=1, checks=results)
    text = render_summary_text(s)
    for r in results:
        assert r.name in text
    assert "PASS" in text
    assert "WARN" in text
    assert "FAIL" in text
    # hint は warning/fail でのみ表示される.
    assert "do this" in text
    assert "fix that" in text
    # pass の detail はあるが hint は元々 None.
    assert "summary:" in text
    assert "exit 18" in text  # failures > 0 なので exit 18.
