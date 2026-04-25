"""Atomic directory swap helper (U6).

`ingest` CLI は `<output_dir>.staging/` にすべての成果物を書き出した後、
本モジュールの `atomic_swap` で `<output_dir>/` へ切替える.

契約 (README「Atomic swap contract」にも記載):

- 同一 FS 上: `os.replace` による POSIX directory-entry 操作で atomic
  (リーダーは旧完全版 or 新完全版のみを観測、中間状態は観測不能).
- クロスデバイス (EXDEV): `<output_dir>.swap-tmp/` に `shutil.copytree` →
  `os.replace` にフォールバック. コピー中の一時状態は sibling dir.
- Durability: best-effort. 親ディレクトリ fsync を行うが電源断耐性は
  保証しない. macOS の `F_FULLFSYNC` は採用していない.
- `--verify-swap` opt-in: staging 側で SHA-256 manifest を生成し、swap 後
  target 側の実ファイルと照合. 同一 FS 上では `os.replace` が inode の
  メタデータ操作なので pre/post hash は構造的に一致する (tautological) —
  主な検出価値は EXDEV fallback 経路および I/O 層の稀な破損.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from lorebook_chunker.errors import AtomicSwapError

logger = logging.getLogger(__name__)

# staging 側で生成し target と riding in するマニフェストのファイル名.
# verify 成功後は `target / MANIFEST_NAME` を削除して出力 dir を clean に保つ.
MANIFEST_NAME = ".swap.manifest.sha256"


def _compute_manifest(root: Path) -> list[tuple[str, int, str]]:
    """Walk `root` for regular files, return stable-sorted `(relpath, size, sha256)` tuples.

    - `relpath` は POSIX 区切り (`/`) で `root` からの相対パス.
    - 既に存在する `MANIFEST_NAME` 自身は除外する (reflexive 比較を避ける).
    - シンボリックリンクは `Path.rglob` の既定挙動に従う (通常の walk).
    - 決定論的な並び順で `sorted(..., key=relpath)` を返す. ハッシュの
      照合は relpath キーで直接突き合わせるため実際には sort 順は
      検出に影響しないが、manifest ファイルのバイト列を再現可能にする
      ために安定ソートする.
    """
    entries: list[tuple[str, int, str]] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        if rel == MANIFEST_NAME:
            continue
        size = file_path.stat().st_size
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        entries.append((rel, size, h.hexdigest()))
    entries.sort(key=lambda t: t[0])
    return entries


def _write_manifest(manifest_path: Path, entries: list[tuple[str, int, str]]) -> None:
    """Write `entries` to `manifest_path` — 1 JSON object per line.

    UTF-8, `ensure_ascii=False`. 行単位でストリーミング parse できる形式.
    """
    with manifest_path.open("w", encoding="utf-8") as f:
        for rel, size, sha in entries:
            line = json.dumps(
                {"path": rel, "size": size, "sha256": sha},
                ensure_ascii=False,
            )
            f.write(line)
            f.write("\n")


def _read_manifest(manifest_path: Path) -> list[tuple[str, int, str]]:
    """Read the manifest file back into the same tuple shape."""
    entries: list[tuple[str, int, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append((obj["path"], int(obj["size"]), obj["sha256"]))
    return entries


def _verify_against_manifest(target: Path, expected: list[tuple[str, int, str]]) -> None:
    """Recompute target-side manifest and raise `AtomicSwapError` on first mismatch.

    失敗時は backup をそのまま残す責務があるため、例外を raise するだけで
    cleanup は呼び元に委ねる (atomic_swap 側で backup を削除しない).
    """
    actual = _compute_manifest(target)
    actual_map = {rel: (size, sha) for rel, size, sha in actual}
    expected_map = {rel: (size, sha) for rel, size, sha in expected}

    # expected にあって actual に無い / hash 不一致 をチェック. 順序は
    # expected の stable sort 順で先頭から.
    for rel, size, sha in expected:
        if rel not in actual_map:
            raise AtomicSwapError(
                f"verify-swap: expected file missing under target: {rel}",
                reason="verify_mismatch",
                path=rel,
                expected=sha,
                actual=None,
            )
        act_size, act_sha = actual_map[rel]
        if act_sha != sha:
            raise AtomicSwapError(
                f"verify-swap: sha256 mismatch at {rel}",
                reason="verify_mismatch",
                path=rel,
                expected=sha,
                actual=act_sha,
            )
    # target 側に expected 外のファイルが混入していないか (staging 完了
    # 後に外部プロセスが書き込んだケース等). 安定ソートして最初の
    # 不一致を報告.
    extras = sorted(set(actual_map) - set(expected_map))
    if extras:
        rel = extras[0]
        act_size, act_sha = actual_map[rel]
        raise AtomicSwapError(
            f"verify-swap: unexpected file under target: {rel}",
            reason="verify_mismatch",
            path=rel,
            expected=None,
            actual=act_sha,
        )


def _best_effort_parent_fsync(target_parent: Path) -> None:
    """Parent-directory fsync. tmpfs 等で未サポートなら warning を吐いて続行."""
    dir_fd: int | None = None
    try:
        dir_fd = os.open(str(target_parent), os.O_DIRECTORY | os.O_RDONLY)
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning(
            "parent-directory fsync unavailable (%s), swap durability best-effort",
            exc,
        )
    finally:
        if dir_fd is not None:
            try:
                os.close(dir_fd)
            except OSError:
                pass


def atomic_swap(target: Path, staging: Path, *, verify: bool = False) -> None:
    """Replace `target` with `staging` atomically on same-fs, copy+replace on EXDEV.

    Steps:
      1. `target.parent` を mkdir (-p 相当).
      2. `verify=True` なら staging を walk して SHA-256 manifest を
         `staging/.swap.manifest.sha256` に書き出す.
      3. 既存 target があれば `target + ".backup"` に rename.
      4. `os.replace(staging, target)` を試行.
         - `EXDEV` → `shutil.copytree(staging, <tmp>)` → `os.replace(<tmp>, target)` →
           staging を rmtree.
         - その他の `OSError` → `AtomicSwapError(reason="replace_failed", ...)`.
      5. 親ディレクトリ fsync (best-effort).
      6. `verify=True` なら manifest と target の実ファイル SHA-256 を照合.
         - mismatch は `AtomicSwapError(reason="verify_mismatch", ...)`.
         - **backup は保持したまま raise** (caller が手動調査可能にするため).
      7. verify 成功時は `target/.swap.manifest.sha256` を削除.
      8. backup を cleanup (rmtree).
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    expected_manifest: list[tuple[str, int, str]] = []
    if verify:
        expected_manifest = _compute_manifest(staging)
        _write_manifest(staging / MANIFEST_NAME, expected_manifest)

    # 3. backup 既存 target を退避.
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(target.name + ".backup")
        if backup.exists():
            # 前回の失敗 run で残ったものを先に掃除. caller が保全したい
            # 場合は atomic_swap を呼ぶ前に rename しておくべき.
            shutil.rmtree(backup)
        os.replace(str(target), str(backup))

    # 4. main swap. os.replace は POSIX の rename 意味論 + 明示的な overwrite.
    try:
        os.replace(str(staging), str(target))
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            # cross-device rename: copytree で bytes コピー → 2 段目 replace で公開.
            tmp = target.with_name(target.name + ".swap-tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(staging, tmp)
            os.replace(str(tmp), str(target))
            shutil.rmtree(staging, ignore_errors=True)
        else:
            raise AtomicSwapError(
                f"atomic swap replace failed: {exc}",
                reason="replace_failed",
                errno=exc.errno or 0,
                source=str(staging),
                target=str(target),
            ) from exc

    # 5. parent-directory fsync (best-effort).
    _best_effort_parent_fsync(target.parent)

    # 6-7. verify (opt-in).
    if verify:
        target_manifest_path = target / MANIFEST_NAME
        # manifest は staging と一緒に ride-in しているはず.
        if not target_manifest_path.exists():  # pragma: no cover - defensive
            raise AtomicSwapError(
                "verify-swap: manifest missing after swap",
                reason="verify_mismatch",
                path=MANIFEST_NAME,
                expected="(present in staging)",
                actual=None,
            )
        # 照合 (mismatch 時は backup を残したまま raise).
        _verify_against_manifest(target, expected_manifest)
        # 成功時のみ manifest を削除.
        try:
            target_manifest_path.unlink()
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning(
                "failed to remove swap manifest under target (%s), leaving in place",
                exc,
            )

    # 8. backup cleanup (verify 失敗時はここに到達しない).
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


__all__ = [
    "atomic_swap",
    "MANIFEST_NAME",
    "_compute_manifest",
]
