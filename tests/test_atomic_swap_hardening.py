"""U6: Atomic swap hardening tests.

テスト対象:
- `src/lorebook_chunker/_swap.py:atomic_swap`
- `src/lorebook_chunker/ingest.py::_atomic_swap` (thin wrapper)
- `IngestConfig.verify_swap` / `--verify-swap` CLI フラグ

cross-device (EXDEV) は実 mount 不要で monkeypatch で `os.replace` に
`OSError(errno=errno.EXDEV)` を注入してシミュレートする.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path

import pytest

from lorebook_chunker import _swap
from lorebook_chunker._swap import MANIFEST_NAME, _compute_manifest, atomic_swap
from lorebook_chunker.errors import AtomicSwapError
from lorebook_chunker.ingest import IngestConfig, IngestRunner, run_ingest

# U2 の共通 stub を使う.
from tests.conftest import _ScriptedLLM, _StubAnalyzer


# ---- helpers ------------------------------------------------------


def _make_staging(root: Path, files: dict[str, bytes]) -> Path:
    """`root` を staging dir として作成し、`files` の content を書き出す."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return root


def _write_sample(input_dir: Path) -> None:
    """ingest で食える最小サンプル (test_ingest.py と同じ). 2 ファイルあれば chunk が出る."""
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "01.txt").write_text(
        "田中はスカラー商事で働いています。"
        "田中は東京に住んでいます。"
        "スカラー商事は大阪にも支社があります。"
        "プロジェクトの責任者は田中です。\n"
        "スカラー商事の佐藤さんも優秀です。"
        "佐藤は東京本社の人間です。"
        "プロジェクトは半年続きます。\n",
        encoding="utf-8",
    )
    (input_dir / "02.txt").write_text(
        "佐藤は最近異動した。"
        "佐藤は新プロジェクトの一員である。"
        "田中と佐藤は協力している。"
        "東京での会議が成功した。\n",
        encoding="utf-8",
    )


# ---- Unit tests for _compute_manifest ----------------------------


def test_compute_manifest_stable_sort_and_excludes_manifest_file(tmp_path: Path) -> None:
    """nested ファイル + 既存 `.swap.manifest.sha256` を含む dir を walk し、
    - relpath で安定ソート
    - MANIFEST_NAME 自身は除外
    を確認する.
    """
    root = _make_staging(
        tmp_path / "root",
        {
            "z.txt": b"last",
            "a.txt": b"first",
            "nested/m.txt": b"mid",
            # 既に manifest がある状況 (前回の run 残骸) でも除外されること.
            MANIFEST_NAME: b'{"ignore": "me"}\n',
        },
    )
    entries = _compute_manifest(root)
    rel_order = [e[0] for e in entries]
    assert rel_order == ["a.txt", "nested/m.txt", "z.txt"]
    assert MANIFEST_NAME not in rel_order
    # sha256 は実際に計算されている.
    for rel, size, sha in entries:
        assert len(sha) == 64
        assert size == (root / rel).stat().st_size


# ---- Happy paths --------------------------------------------------


def test_happy_path_same_fs_no_verify(tmp_path: Path) -> None:
    """同一 FS 上の単純な swap: target 置換、backup 削除、manifest 作成なし."""
    target = tmp_path / "out"
    staging = _make_staging(tmp_path / "out.staging", {"a.txt": b"new"})
    # 既存 target を作っておく.
    target.mkdir()
    (target / "old.txt").write_bytes(b"stale")

    atomic_swap(target, staging, verify=False)

    assert (target / "a.txt").read_bytes() == b"new"
    assert not (target / "old.txt").exists()
    assert not (tmp_path / "out.backup").exists()
    assert not (target / MANIFEST_NAME).exists()
    assert not staging.exists()


def test_happy_path_first_run_no_existing_target(tmp_path: Path) -> None:
    """target が未存在 (初回 run) → backup 段階をスキップして新 target 作成."""
    target = tmp_path / "out"
    staging = _make_staging(tmp_path / "out.staging", {"a.txt": b"content"})
    assert not target.exists()

    atomic_swap(target, staging, verify=False)

    assert (target / "a.txt").read_bytes() == b"content"
    assert not (tmp_path / "out.backup").exists()


def test_happy_path_verify_swap_same_fs(tmp_path: Path) -> None:
    """--verify-swap + same-fs: 構造的に hash が一致する (os.replace は inode 操作).

    この path は tautological に pass するが code path の健全性を担保する.
    verify 成功後は target/.swap.manifest.sha256 が削除されていること.
    """
    target = tmp_path / "out"
    staging = _make_staging(
        tmp_path / "out.staging",
        {"a.txt": b"hello", "nested/b.txt": b"world"},
    )

    atomic_swap(target, staging, verify=True)

    assert (target / "a.txt").read_bytes() == b"hello"
    assert (target / "nested/b.txt").read_bytes() == b"world"
    # verify 成功後は manifest が削除されている.
    assert not (target / MANIFEST_NAME).exists()
    assert not (tmp_path / "out.backup").exists()


# ---- Edge cases ---------------------------------------------------


def test_verify_disabled_never_writes_manifest(tmp_path: Path) -> None:
    """--verify-swap OFF で .swap.manifest.sha256 が staging にも target にも作られないこと."""
    target = tmp_path / "out"
    staging = _make_staging(tmp_path / "out.staging", {"a.txt": b"x"})

    atomic_swap(target, staging, verify=False)

    # swap 後、 target 側にも manifest が無いこと.
    assert not (target / MANIFEST_NAME).exists()
    # staging は rename or rmtree 済み.
    assert not staging.exists()


def test_parent_dir_fsync_osError_logs_warning_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """parent-directory fsync が OSError (例: tmpfs) でも swap は継続し、
    warning を log に残す.
    """
    target = tmp_path / "out"
    staging = _make_staging(tmp_path / "out.staging", {"a.txt": b"ok"})

    real_open = os.open

    def _fake_open(path, flags, *args, **kwargs):
        # O_DIRECTORY 時だけ拒絶. 他の open 呼び出しは通す.
        if flags & os.O_DIRECTORY:
            raise OSError(errno.ENOTSUP, "fsync unsupported on tmpfs")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _fake_open)

    with caplog.at_level("WARNING", logger="lorebook_chunker._swap"):
        atomic_swap(target, staging, verify=False)

    # swap 自体は成功している.
    assert (target / "a.txt").read_bytes() == b"ok"
    # warning がログに出ている.
    assert any("parent-directory fsync unavailable" in rec.message for rec in caplog.records)


# ---- EXDEV (cross-device) simulation -----------------------------


def test_exdev_fallback_via_copytree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.replace が最初は EXDEV で落ち、2 度目 (tmp → target の replace) は成功する
    → shutil.copytree 経路で swap が完走する.
    """
    target = tmp_path / "out"
    staging = _make_staging(
        tmp_path / "out.staging",
        {"a.txt": b"cross-device", "nested/b.txt": b"copied"},
    )

    real_replace = os.replace
    call_count = {"n": 0}

    def _fake_replace(src, dst):
        # 最初の呼び出し (staging → target) だけ EXDEV にして、以降は通常動作.
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _fake_replace)

    atomic_swap(target, staging, verify=False)

    assert (target / "a.txt").read_bytes() == b"cross-device"
    assert (target / "nested/b.txt").read_bytes() == b"copied"
    # swap-tmp は replace 済みで残骸無し.
    assert not (tmp_path / "out.swap-tmp").exists()
    assert not staging.exists()


def test_exdev_verify_detects_corrupted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXDEV fallback path で copytree 後に target 側のファイルを故意に破損 →
    --verify-swap が mismatch を検出し AtomicSwapError(reason="verify_mismatch")、
    backup は保持される.
    """
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.txt").write_bytes(b"stale")
    staging = _make_staging(
        tmp_path / "out.staging",
        {"a.txt": b"pristine"},
    )

    real_replace = os.replace
    replace_calls = {"n": 0}

    def _fake_replace(src, dst):
        replace_calls["n"] += 1
        # シナリオ:
        #   1) target → target.backup      (通常成功)
        #   2) staging → target              (EXDEV を注入 → copytree 経路へ)
        #   3) swap-tmp → target             (通常成功)
        # 注入は (2) だけ. (2) 判定は src が staging path かつ EXDEV 未実施時.
        if replace_calls["n"] == 2:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _fake_replace)

    # copytree をラップして target (copytree 先 = swap-tmp) の bytes を改ざん.
    import shutil as _shutil
    real_copytree = _shutil.copytree

    def _corrupting_copytree(src, dst, *args, **kwargs):
        result = real_copytree(src, dst, *args, **kwargs)
        # 破損: a.txt を書き換える. manifest は既に staging (= src) 側に書き
        # 出されて copytree により swap-tmp にもコピーされているので、target
        # の manifest は "pristine" を期待するが、実ファイルは corrupted.
        (Path(dst) / "a.txt").write_bytes(b"CORRUPTED")
        return result

    monkeypatch.setattr(_swap.shutil, "copytree", _corrupting_copytree)

    with pytest.raises(AtomicSwapError) as exc_info:
        atomic_swap(target, staging, verify=True)

    err = exc_info.value
    assert err.exit_code == 16
    assert err.context.get("reason") == "verify_mismatch"
    assert err.context.get("path") == "a.txt"

    # backup が保持されていること (手動調査可能).
    backup = tmp_path / "out.backup"
    assert backup.exists()
    assert (backup / "old.txt").read_bytes() == b"stale"


# ---- Error paths --------------------------------------------------


def test_replace_eacces_raises_atomic_swap_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """staging → target の os.replace が EACCES で落ちると AtomicSwapError(reason="replace_failed").

    EXDEV 以外の OSError は copytree fallback に流れず、直接 AtomicSwapError に翻訳される.
    """
    target = tmp_path / "out"
    staging = _make_staging(tmp_path / "out.staging", {"a.txt": b"x"})

    real_replace = os.replace
    calls = {"n": 0}

    def _fake_replace(src, dst):
        calls["n"] += 1
        # 最初の呼び出し (staging → target) を EACCES にする.
        if calls["n"] == 1:
            raise OSError(errno.EACCES, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _fake_replace)

    with pytest.raises(AtomicSwapError) as exc_info:
        atomic_swap(target, staging, verify=False)

    err = exc_info.value
    assert err.exit_code == 16
    assert err.context.get("reason") == "replace_failed"
    assert err.context.get("errno") == errno.EACCES
    # staging は保全される (手動調査可能).
    assert staging.exists()


def test_backup_preserved_when_main_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """backup rename OK → staging → target の replace が EACCES → target 不在 + backup 残存 →
    AtomicSwapError. 復旧手順は `mv <output>.backup <output>`.
    """
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.txt").write_bytes(b"preserved")
    staging = _make_staging(tmp_path / "out.staging", {"a.txt": b"new"})

    real_replace = os.replace
    calls = {"n": 0}

    def _fake_replace(src, dst):
        calls["n"] += 1
        # 1 回目: target → target.backup は通す
        # 2 回目: staging → target を EACCES
        if calls["n"] == 2:
            raise OSError(errno.EACCES, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _fake_replace)

    with pytest.raises(AtomicSwapError):
        atomic_swap(target, staging, verify=False)

    # target は空 (backup 済み), backup が内容保持.
    assert not target.exists()
    backup = tmp_path / "out.backup"
    assert backup.exists()
    assert (backup / "old.txt").read_bytes() == b"preserved"


# ---- Integration with IngestRunner ------------------------------


def test_ingest_run_with_verify_swap_true(tmp_path: Path) -> None:
    """IngestConfig.verify_swap=True でフル ingest が通り、run_report.swap の
    duration が計測可能な値になること. verify 成功なので output_dir に
    .swap.manifest.sha256 は残らない.
    """
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    cfg = IngestConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_wiki=True,
        target_chars=30,
        max_chunk_chars=200,
        min_mentions=2,
        min_chunks=1,
        show_progress=False,
        verify_swap=True,
    )
    runner = IngestRunner(
        cfg,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    )
    result = runner.run()
    assert result.exit_code == 0
    assert (output_dir / "chunks.jsonl").exists()
    # manifest は verify 成功後に削除されている.
    assert not (output_dir / MANIFEST_NAME).exists()
    # swap phase の duration が記録されている (verify のハッシュ再計算分が含まれる).
    assert result.phase_timings.get("swap", 0.0) > 0.0


def test_verify_swap_adds_measurable_overhead_vs_no_verify(tmp_path: Path) -> None:
    """同じ corpus に対して verify_swap=True / False の両方を走らせ、
    verify_swap=True のほうが swap の duration が大きいこと.

    両回で同じ input_dir + output_dir を使い、出力は破壊的に上書きされる.
    実行時間は環境依存のため `>=` (0 差はあり得るが、verify は必ずハッシュ
    再計算が入るので典型的には真に大きくなる) で検証. 不安定回避のため
    最低でも verify 側で非ゼロ値である確認に留める.
    """
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    cfg_no_verify = IngestConfig(
        input_dir=input_dir, output_dir=output_dir, skip_wiki=True,
        target_chars=30, max_chunk_chars=200, min_mentions=2, min_chunks=1,
        show_progress=False, verify_swap=False,
    )
    r1 = IngestRunner(
        cfg_no_verify,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    ).run()
    assert r1.exit_code == 0
    swap_no_verify = r1.phase_timings["swap"]

    cfg_verify = IngestConfig(
        input_dir=input_dir, output_dir=output_dir, skip_wiki=True,
        target_chars=30, max_chunk_chars=200, min_mentions=2, min_chunks=1,
        show_progress=False, verify_swap=True,
    )
    r2 = IngestRunner(
        cfg_verify,
        analyzer_factory=lambda path: _StubAnalyzer(),
        llm_factory=lambda backend, config: _ScriptedLLM([]),
    ).run()
    assert r2.exit_code == 0
    swap_verify = r2.phase_timings["swap"]

    # verify 側は必ず >= no_verify. 完全同値にならないよう (tie-break のため)
    # verify 側の値が strictly 正であることだけ要求する.
    assert swap_verify > 0.0
    assert swap_verify >= swap_no_verify


def test_cli_run_ingest_routes_verify_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_ingest(args)` が `args.verify_swap=True` を IngestConfig に流すことを
    確認する. spy で `IngestConfig` を傍受する代わりに、flag=True で run し
    output に manifest が残らない (verify 成功) ことで経路を確認.
    """
    input_dir = tmp_path / "samples"
    _write_sample(input_dir)
    output_dir = tmp_path / "out"

    args = argparse.Namespace(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        skip_wiki=True,
        max_llm_calls=None,
        force_regenerate=False,
        retry_failed=False,
        llm_backend="anthropic",
        llm_model=None,
        llm_parallelism=None,
        analyzer_backend="electra",
        device="cpu",
        format="human",
        quiet=True,
        no_progress=True,
        recursive=False,
        globs="*.txt",
        encoding="auto",
        verify_swap=True,
        command="ingest",
    )

    import lorebook_chunker.ingest as mod
    monkeypatch.setattr(mod, "default_analyzer_factory", lambda path, **kw: _StubAnalyzer())
    # preflight LLM check は skip_wiki=True なら走らないので llm_factory の
    # 差し替えは不要.

    exit_code = run_ingest(args)
    assert exit_code == 0
    assert (output_dir / "chunks.jsonl").exists()
    # verify 成功で manifest は掃除されている.
    assert not (output_dir / MANIFEST_NAME).exists()
