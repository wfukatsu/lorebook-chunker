"""U2: `--recursive` / `--glob` 入力探索オプションのテスト.

`_collect_input_files` の振る舞い (recursive / multi-pattern / dedup / sort) と
CLI boundary (`run_ingest` → ConfigError) を組み合わせて検証する.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lorebook_chunker.errors import ConfigError
from lorebook_chunker.ingest import (
    IngestRunner,
    _collect_input_files,
    _parse_globs_arg,
    _validate_glob_patterns,
)


# ---- _parse_globs_arg --------------------------------------------------


def test_parse_globs_default_when_none() -> None:
    assert _parse_globs_arg(None) == ("*.txt",)


def test_parse_globs_splits_comma() -> None:
    assert _parse_globs_arg("*.txt,*.md") == ("*.txt", "*.md")


def test_parse_globs_trims_whitespace() -> None:
    assert _parse_globs_arg(" *.txt , *.md ") == ("*.txt", "*.md")


def test_parse_globs_drops_empty_entries() -> None:
    # "," のみ / 空文字混在 → tuple() → 後段の _validate で ConfigError
    assert _parse_globs_arg(",,") == ()
    assert _parse_globs_arg("*.txt,,*.md") == ("*.txt", "*.md")


# ---- _validate_glob_patterns -----------------------------------------


def test_validate_empty_tuple_raises() -> None:
    with pytest.raises(ConfigError) as ei:
        _validate_glob_patterns(())
    assert ei.value.exit_code == 2
    assert ei.value.context.get("reason") == "invalid_glob"


def test_validate_empty_string_raises() -> None:
    with pytest.raises(ConfigError) as ei:
        _validate_glob_patterns(("",))
    assert ei.value.context.get("reason") == "invalid_glob"
    assert ei.value.context.get("pattern") == ""


def test_validate_absolute_path_raises() -> None:
    with pytest.raises(ConfigError) as ei:
        _validate_glob_patterns(("/etc/**",))
    assert ei.value.context.get("reason") == "invalid_glob"
    assert ei.value.context.get("pattern") == "/etc/**"


def test_validate_relative_patterns_ok() -> None:
    # 例外が上がらないこと (returns None)
    _validate_glob_patterns(("*.txt",))
    _validate_glob_patterns(("*.txt", "*.md"))
    _validate_glob_patterns(("sub/*.txt",))  # subdir pattern は許容


# ---- _collect_input_files: happy paths --------------------------------


def test_default_non_recursive_finds_top_level_only(tmp_path: Path, tmp_samples_dir) -> None:
    """Regression: default (flag 無し) で top-level *.txt のみ検出."""
    root = tmp_samples_dir(
        tmp_path,
        {
            "a.txt": "hello",
            "b.txt": "world",
            "sub/deep.txt": "nested",  # 再帰指定なしなら除外
        },
    )
    found = _collect_input_files(root)
    names = [p.name for p in found]
    assert names == ["a.txt", "b.txt"]


def test_recursive_on_flat_dir_matches_top_level(tmp_path: Path, tmp_samples_dir) -> None:
    root = tmp_samples_dir(tmp_path, {"a.txt": "a", "b.txt": "b", "c.txt": "c"})
    non_recursive = _collect_input_files(root, recursive=False)
    recursive = _collect_input_files(root, recursive=True)
    assert [p.name for p in non_recursive] == ["a.txt", "b.txt", "c.txt"]
    assert [p.resolve() for p in recursive] == [p.resolve() for p in non_recursive]


def test_recursive_nested_dir_finds_subfiles(tmp_path: Path, tmp_samples_dir) -> None:
    root = tmp_samples_dir(
        tmp_path,
        {
            "root.txt": "r",
            "sub/a.txt": "a",
            "sub/deep/b.txt": "b",
        },
    )
    found = _collect_input_files(root, recursive=True)
    rels = sorted(str(p.relative_to(root)) for p in found)
    # path separator は OS 依存 (POSIX では '/')
    assert rels == sorted([
        "root.txt",
        str(Path("sub/a.txt")),
        str(Path("sub/deep/b.txt")),
    ])


def test_glob_md_only(tmp_path: Path, tmp_samples_dir) -> None:
    root = tmp_samples_dir(
        tmp_path,
        {
            "a.txt": "t",
            "b.md": "m",
            "c.md": "m",
        },
    )
    found = _collect_input_files(root, globs=("*.md",))
    assert [p.name for p in found] == ["b.md", "c.md"]


def test_glob_multiple_patterns_sorted_no_dup(tmp_path: Path, tmp_samples_dir) -> None:
    root = tmp_samples_dir(
        tmp_path,
        {
            "a.txt": "t",
            "b.md": "m",
            "c.txt": "t",
        },
    )
    found = _collect_input_files(root, globs=("*.txt", "*.md"))
    # dedupe + stable sort
    assert [p.name for p in found] == ["a.txt", "b.md", "c.txt"]
    # no duplicates
    assert len(found) == len({p.resolve() for p in found})


# ---- _collect_input_files: edge cases --------------------------------


def test_zero_matches_returns_empty_list(tmp_path: Path, tmp_samples_dir) -> None:
    """0 件は空 list を返す (caller 側で ConfigError 化する). 同様に recursive=True も."""
    root = tmp_samples_dir(tmp_path, {"a.md": "m"})  # *.txt は 0 件
    assert _collect_input_files(root, globs=("*.txt",)) == []
    assert _collect_input_files(root, globs=("*.txt",), recursive=True) == []


def test_overlapping_globs_dedupe_by_resolved_path(tmp_path: Path, tmp_samples_dir) -> None:
    """同じファイルが複数パターンにマッチしても 1 件になる."""
    root = tmp_samples_dir(tmp_path, {"a.txt": "t", "b.txt": "t"})
    found = _collect_input_files(root, globs=("*.txt", "a.*", "*.txt"))
    assert [p.name for p in found] == ["a.txt", "b.txt"]
    assert len({p.resolve() for p in found}) == 2


def test_deeply_nested_many_files_completes(tmp_path: Path, tmp_samples_dir) -> None:
    """ネスト 5 階層 / 100 件超でも rglob が完走する (stack overflow しない)."""
    content_map: dict[str, bytes | str] = {}
    for i in range(120):
        depth = i % 5
        rel_parts = [f"d{j}" for j in range(depth)] + [f"f{i:03d}.txt"]
        content_map["/".join(rel_parts)] = f"body {i}"
    root = tmp_samples_dir(tmp_path, content_map)
    found = _collect_input_files(root, recursive=True)
    assert len(found) == 120
    # sort stability: Path 比較で strictly 昇順
    assert found == sorted(found)


# ---- _collect_input_files: error paths ------------------------------


def test_absolute_glob_raises_config_error(tmp_path: Path, tmp_samples_dir) -> None:
    root = tmp_samples_dir(tmp_path, {"a.txt": "t"})
    with pytest.raises(ConfigError) as ei:
        _collect_input_files(root, globs=("/etc/**",))
    assert ei.value.exit_code == 2
    assert ei.value.context.get("reason") == "invalid_glob"


def test_empty_glob_raises_config_error(tmp_path: Path, tmp_samples_dir) -> None:
    root = tmp_samples_dir(tmp_path, {"a.txt": "t"})
    with pytest.raises(ConfigError) as ei:
        _collect_input_files(root, globs=("",))
    assert ei.value.exit_code == 2
    assert ei.value.context.get("reason") == "invalid_glob"


# ---- IngestRunner: zero-input → ConfigError (exit 2) ------------------


def test_runner_zero_matches_raises_config_error(
    tmp_path: Path, tmp_samples_dir, ingest_config_factory
) -> None:
    """既存 U1 contract: 0 件は ConfigError(reason=no_input_files, exit=2)."""
    root = tmp_samples_dir(tmp_path, {"a.md": "m"})
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=root,
        output_dir=tmp_path / "out",
        globs=("*.txt",),
    )
    result = IngestRunner(
        cfg,
        analyzer_factory=lambda path: None,  # type: ignore[return-value]
        llm_factory=lambda backend, config: None,  # type: ignore[return-value]
    ).run()
    assert result.exit_code == 2
    assert result.errors, "errors should carry ConfigError payload"
    err = result.errors[0]
    assert err["class"] == "ConfigError"
    assert err["context"].get("reason") == "no_input_files"
    assert "*.txt" in err["context"].get("globs", [])


def test_runner_invalid_glob_raises_config_error(
    tmp_path: Path, tmp_samples_dir, ingest_config_factory
) -> None:
    root = tmp_samples_dir(tmp_path, {"a.txt": "t"})
    cfg = ingest_config_factory(
        tmp_path,
        input_dir=root,
        output_dir=tmp_path / "out",
        globs=("",),  # invalid
    )
    result = IngestRunner(
        cfg,
        analyzer_factory=lambda path: None,  # type: ignore[return-value]
        llm_factory=lambda backend, config: None,  # type: ignore[return-value]
    ).run()
    assert result.exit_code == 2
    assert result.errors
    assert result.errors[0]["class"] == "ConfigError"
    assert result.errors[0]["context"].get("reason") == "invalid_glob"


# ---- CLI help integration --------------------------------------------


def test_ingest_help_mentions_recursive_and_glob() -> None:
    """`ingest --help` に `--recursive` と `--glob` が日本語説明付きで表示される."""
    proc = subprocess.run(
        [sys.executable, "-m", "lorebook_chunker.cli", "ingest", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    help_text = proc.stdout
    assert "--recursive" in help_text
    assert "--glob" in help_text
    # 日本語説明が乗っている
    assert "再帰" in help_text
    assert "glob" in help_text or "パターン" in help_text
