"""`lint` CLI: コーパス健全性の致命/警告/情報 3 段階チェック."""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.sparse import csr_matrix

from chunking.tfidf import TfidfBuilder

logger = logging.getLogger(__name__)

FATAL = "致命"
WARNING = "警告"
INFO = "情報"

DEFAULT_LINT_PAIRWISE_THRESHOLD = 2000
DEFAULT_DUPLICATE_COSINE = 0.95
DEFAULT_DEGENERATE_L2 = 1e-6
DEFAULT_DEGENERATE_NNZ = 3
DEFAULT_LEVENSHTEIN_RATIO = 0.85


@dataclass
class LintFinding:
    severity: str
    category: str
    message: str
    detail: str = ""


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)

    def add(self, severity: str, category: str, message: str, detail: str = "") -> None:
        self.findings.append(LintFinding(severity=severity, category=category, message=message, detail=detail))

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)


def run_lint_impl(
    output_dir: Path,
    *,
    pairwise_threshold: int = DEFAULT_LINT_PAIRWISE_THRESHOLD,
    duplicate_cosine: float = DEFAULT_DUPLICATE_COSINE,
    degenerate_l2: float = DEFAULT_DEGENERATE_L2,
    degenerate_nnz: int = DEFAULT_DEGENERATE_NNZ,
    levenshtein_ratio: float = DEFAULT_LEVENSHTEIN_RATIO,
    target_chunk_chars: int = 500,
    max_chunk_chars: int = 1500,
) -> LintReport:
    report = LintReport()

    analyzer_path = output_dir / "analyzer.json"
    vocab_path = output_dir / "vocab.npz"
    chunks_path = output_dir / "chunks.jsonl"

    missing_required = [p for p in (analyzer_path, vocab_path, chunks_path) if not p.exists()]
    if missing_required:
        report.add(
            FATAL,
            "missing_artifact",
            "必要成果物が存在しません",
            detail=", ".join(str(p) for p in missing_required),
        )
        return report

    # analyzer.json: JSON として読める + strict_match/compat_match の構造を持つ
    try:
        with analyzer_path.open("r", encoding="utf-8") as f:
            analyzer_json = json.load(f)
    except json.JSONDecodeError as e:
        report.add(FATAL, "schema", "analyzer.json が JSON として壊れています", detail=str(e))
        return report
    if "strict_match" not in analyzer_json or "compat_match" not in analyzer_json:
        report.add(
            FATAL,
            "schema",
            "analyzer.json に strict_match / compat_match キーがありません",
        )
        return report

    # vocab.npz
    try:
        arts = TfidfBuilder.load(vocab_path)
    except Exception as e:
        report.add(FATAL, "schema", "vocab.npz 読み込み失敗", detail=str(e))
        return report

    # chunks.jsonl
    chunks = list(_load_chunks(chunks_path))
    if not chunks:
        report.add(FATAL, "empty_corpus", "chunks.jsonl が空です")
        return report
    if arts.matrix.shape[0] != len(chunks):
        report.add(
            FATAL,
            "schema",
            f"chunks.jsonl ({len(chunks)}) と vocab.npz matrix ({arts.matrix.shape[0]}) 行数不一致",
        )
        return report

    # --- 警告レベル ---

    # チャンク長チェック
    for chunk in chunks:
        length = len(chunk.get("text", ""))
        if length == 0:
            report.add(WARNING, "empty_chunk", f"空チャンク chunk_id={chunk['chunk_id']}")
        if length > max_chunk_chars:
            report.add(
                WARNING,
                "over_max_chunk",
                f"max_chunk_chars={max_chunk_chars} 超過 chunk_id={chunk['chunk_id']} ({length} chars)",
            )
        elif length > 2 * target_chunk_chars:
            report.add(
                WARNING,
                "over_2x_target",
                f"目標の 2 倍超 chunk_id={chunk['chunk_id']} ({length} chars > {2 * target_chunk_chars})",
            )
        if length > 0 and _looks_single_sentence(chunk.get("text", "")):
            report.add(
                INFO,
                "single_sentence",
                f"1 文のみチャンク chunk_id={chunk['chunk_id']}",
            )

    # TF-IDF 縮退
    degenerate_chunks = _detect_degenerate_rows(
        arts.matrix, l2_threshold=degenerate_l2, nnz_threshold=degenerate_nnz
    )
    for idx in degenerate_chunks:
        report.add(
            WARNING,
            "degenerate_vector",
            f"TF-IDF 縮退 (L2<{degenerate_l2} または nnz<{degenerate_nnz}) chunk_id={chunks[idx]['chunk_id']}",
        )

    # 重複検出 (pairwise; スケール制限を守る)
    n = arts.matrix.shape[0]
    if n <= pairwise_threshold:
        dup_pairs = _detect_duplicates(arts.matrix, threshold=duplicate_cosine)
        for i, j, cos in dup_pairs:
            report.add(
                WARNING,
                "duplicate",
                f"ほぼ重複 ({chunks[i]['chunk_id']}, {chunks[j]['chunk_id']}) cos={cos:.3f}",
            )
    else:
        report.add(
            INFO,
            "pairwise_skipped",
            f"pairwise 比較スキップ (n={n} > threshold {pairwise_threshold})",
            detail="コサイン重複検出と表記近似エンティティ検出をスキップしました",
        )

    # エンティティ wiki のチェック
    entities_dir = output_dir / "entities"
    manifest_path = entities_dir / "manifest.json"
    success_entries = _load_success_entries(manifest_path)
    chunk_ids_present = {c["chunk_id"] for c in chunks}
    for entry in success_entries:
        referenced = set(entry.get("chunk_ids", []))
        if not (referenced & chunk_ids_present):
            report.add(
                WARNING,
                "orphan_wiki",
                f"孤立 wiki: {entry.get('entity_name')} ({entry.get('ner_label')}) の参照 chunk が全て消失",
            )

    # wiki 機能失効: wiki < 3 AND chunks < 20
    successful_count = len(success_entries)
    if successful_count < 3 and len(chunks) < 20:
        report.add(
            WARNING,
            "wiki_disabled",
            f"wiki 生成数 {successful_count} かつチャンク数 {len(chunks)} < 20: "
            "コーパス規模が小さく wiki 機能が事実上失効しています",
        )

    # 表記近似エンティティ対 (pairwise)
    if entities_dir.exists() and len(success_entries) <= pairwise_threshold:
        pairs = _detect_similar_names(success_entries, ratio=levenshtein_ratio)
        for a, b, r in pairs:
            report.add(
                INFO,
                "similar_entity",
                f"表記近似候補: {a} <-> {b} (ratio={r:.3f})",
            )

    return report


# ---- helpers ---------------------------------------------------


def _load_chunks(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _looks_single_sentence(text: str) -> bool:
    """句点「。」が末尾以外に現れなければ 1 文扱い."""
    count = text.count("。")
    return count <= 1


def _detect_degenerate_rows(
    matrix: csr_matrix,
    *,
    l2_threshold: float,
    nnz_threshold: int,
) -> list[int]:
    out: list[int] = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz < nnz_threshold:
            out.append(i)
            continue
        norm = float(np.sqrt(row.multiply(row).sum()))
        if norm < l2_threshold:
            out.append(i)
    return out


def _detect_duplicates(
    matrix: csr_matrix, *, threshold: float
) -> list[tuple[int, int, float]]:
    """L2-normalized 前提のコサイン類似度で threshold を超えるペアを検出."""
    # TfidfVectorizer の norm='l2' デフォルトで matrix は既に L2 正規化済み
    sim = matrix @ matrix.T
    sim_dense = sim.toarray()
    n = sim_dense.shape[0]
    np.fill_diagonal(sim_dense, 0.0)
    out: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_dense[i, j] > threshold:
                out.append((i, j, float(sim_dense[i, j])))
    return out


def _load_success_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    entries = data.get("entries", {}) or {}
    return [e for e in entries.values() if e.get("status") == "success"]


def _detect_similar_names(
    entries: list[dict[str, Any]], *, ratio: float
) -> list[tuple[str, str, float]]:
    """表記近似検出. Levenshtein 比 > ratio のペアを返す (O(N^2))."""
    out: list[tuple[str, str, float]] = []
    names = [(f"{e.get('ner_label')}__{e.get('entity_name')}", e.get("entity_name", "")) for e in entries]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_id, a_name = names[i]
            b_id, b_name = names[j]
            r = difflib.SequenceMatcher(None, a_name, b_name).ratio()
            if r > ratio:
                out.append((a_id, b_id, r))
    return out


# ---- CLI ------------------------------------------------------------


def run_lint(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    report = run_lint_impl(output_dir)
    _print_human_report(report, output_dir)
    _write_lint_md(output_dir / "lint.md", report)
    return 0 if report.count(FATAL) == 0 else 1


def _print_human_report(report: LintReport, output_dir: Path) -> None:
    fatal = report.count(FATAL)
    warn = report.count(WARNING)
    info = report.count(INFO)
    print(f"Lint report for {output_dir}: 致命 {fatal} / 警告 {warn} / 情報 {info}", file=sys.stderr)
    for f in report.findings:
        print(f"[{f.severity}] {f.category}: {f.message}", file=sys.stderr)
        if f.detail:
            print(f"    {f.detail}", file=sys.stderr)


def _write_lint_md(path: Path, report: LintReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Lint Report", ""]
    lines.append(
        f"- 致命: {report.count(FATAL)}"
    )
    lines.append(f"- 警告: {report.count(WARNING)}")
    lines.append(f"- 情報: {report.count(INFO)}")
    lines.append("")
    for severity in (FATAL, WARNING, INFO):
        items = [f for f in report.findings if f.severity == severity]
        if not items:
            continue
        lines.append(f"## {severity}")
        lines.append("")
        lines.append("| カテゴリ | メッセージ | 詳細 |")
        lines.append("|---|---|---|")
        for f in items:
            detail = f.detail.replace("|", "\\|") if f.detail else ""
            message = f.message.replace("|", "\\|")
            lines.append(f"| {f.category} | {message} | {detail} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
