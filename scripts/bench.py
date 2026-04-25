#!/usr/bin/env python3
"""RAG 検索評価ベンチマーク (chunking params regression / チューニング用途).

本スクリプトは **自前 corpus のチューニング / regression 検出用途** です.
ツール間の quality 比較は対象外. 外部データセット (JQaRA, MIRACL-ja 等) との
比較は別 release で検討予定です.

Usage:
    python scripts/bench.py <corpus_dir> \\
        --configs NAME1=chunk:512,overlap:64,wiki:on \\
        --configs NAME2=chunk:256,overlap:32,wiki:off \\
        --qrels samples/qrels.jsonl \\
        [--out bench_out/] [--json bench_report.json]

Exit codes:
    0   : success
    2   : ConfigError (invalid config spec / insufficient configs / missing qrels
          / ranx 未導入 / empty qrels)
    3-17: ingest-phase errors propagated from the first failing config
          (see README exit-code table)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `lorebook_chunker` を src レイアウトのまま呼び出せるよう、pip install 前提の
# console script 的な使い方 (module import) と直接実行の両方に対応する.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lorebook_chunker.errors import ConfigError  # noqa: E402
from lorebook_chunker.ingest import IngestConfig, IngestRunner  # noqa: E402
from lorebook_chunker.query import run_query_impl  # noqa: E402


METRICS: tuple[str, ...] = ("recall@5", "recall@10", "mrr@10", "ndcg@10")


# ---- Config spec parsing --------------------------------------------------


@dataclass
class BenchConfig:
    """1 つの `--configs NAME=key:val,...` 引数から得られる構造化設定."""

    name: str
    chunk_size: int
    overlap: int
    skip_wiki: bool


def parse_config_spec(spec: str) -> BenchConfig:
    """`NAME=chunk:512,overlap:64,wiki:on` 形式を BenchConfig に parse する.

    - `chunk:<int>` / `overlap:<int>` / `wiki:on|off` の 3 キーを認識.
    - 未指定のキーはデフォルト値 (chunk=500, overlap=100, wiki=off) で埋める.
    - 未知のキー / 不正値 / 空 NAME は `ConfigError(reason="invalid_config_spec")`.
    """
    name, sep, params = spec.partition("=")
    if not sep or not name or not params:
        raise ConfigError(
            f"invalid --configs spec: {spec!r} (expected NAME=key:val,...)",
            reason="invalid_config_spec",
            spec=spec,
        )
    chunk_size: int | None = None
    overlap: int | None = None
    skip_wiki: bool | None = None
    for kv in params.split(","):
        kv = kv.strip()
        if not kv:
            continue
        key, ksep, value = kv.partition(":")
        key = key.strip()
        value = value.strip()
        if not ksep or not key or not value:
            raise ConfigError(
                f"invalid key:value pair in --configs: {kv!r}",
                reason="invalid_config_spec",
                spec=spec,
                pair=kv,
            )
        if key == "chunk":
            try:
                chunk_size = int(value)
            except ValueError as e:
                raise ConfigError(
                    f"chunk must be integer, got {value!r}",
                    reason="invalid_config_spec",
                    spec=spec,
                    key=key,
                    value=value,
                ) from e
        elif key == "overlap":
            try:
                overlap = int(value)
            except ValueError as e:
                raise ConfigError(
                    f"overlap must be integer, got {value!r}",
                    reason="invalid_config_spec",
                    spec=spec,
                    key=key,
                    value=value,
                ) from e
        elif key == "wiki":
            if value == "on":
                skip_wiki = False
            elif value == "off":
                skip_wiki = True
            else:
                raise ConfigError(
                    f"wiki must be 'on' or 'off', got {value!r}",
                    reason="invalid_config_spec",
                    spec=spec,
                    key=key,
                    value=value,
                )
        else:
            raise ConfigError(
                f"unknown key in --configs: {key!r}",
                reason="unknown_config_key",
                spec=spec,
                key=key,
            )
    return BenchConfig(
        name=name,
        chunk_size=chunk_size if chunk_size is not None else 500,
        overlap=overlap if overlap is not None else 100,
        skip_wiki=skip_wiki if skip_wiki is not None else True,
    )


# ---- Qrels I/O ------------------------------------------------------------


@dataclass
class QrelEntry:
    qid: str
    query: str
    relevant: dict[str, int]  # chunk_id -> grade


def load_qrels(path: Path) -> list[QrelEntry]:
    """JSONL qrels を読み込む.

    各行: ``{"qid": "q1", "query": "...", "relevant": [{"chunk_id": "...", "grade": 2}, ...]}``.

    空ファイル / qid 0 件 → `ConfigError(reason="empty_qrels")`.
    """
    if not path.exists():
        raise ConfigError(
            f"qrels file not found: {path}",
            reason="qrels_not_found",
            path=str(path),
        )
    entries: list[QrelEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ConfigError(
                    f"qrels line {lineno} is not valid JSON: {e}",
                    reason="qrels_parse_error",
                    path=str(path),
                    lineno=lineno,
                ) from e
            qid = obj.get("qid")
            query = obj.get("query")
            relevant = obj.get("relevant", [])
            if not qid or not query:
                raise ConfigError(
                    f"qrels line {lineno} missing qid/query",
                    reason="qrels_parse_error",
                    path=str(path),
                    lineno=lineno,
                )
            rel_map: dict[str, int] = {}
            for item in relevant:
                cid = item.get("chunk_id")
                grade = int(item.get("grade", 1))
                if cid:
                    rel_map[cid] = grade
            entries.append(QrelEntry(qid=str(qid), query=str(query), relevant=rel_map))
    if not entries:
        raise ConfigError(
            f"qrels is empty: {path}",
            reason="empty_qrels",
            path=str(path),
        )
    return entries


# ---- ranx guard -----------------------------------------------------------


def _ensure_ranx() -> Any:
    """ranx を lazy import. 未導入時は `ConfigError` (exit 2) を出す.

    import 自体は top-level で wrap する代わりに、本関数を CLI invocation 時
    にだけ呼ぶことで、`scripts/bench.py` を `ranx` 未導入環境でも import /
    parse 可能に保つ (テスト skip 判定にも使える).
    """
    try:
        import ranx  # type: ignore
    except ImportError as e:
        raise ConfigError(
            "ranx is not installed; RAG evaluation benchmark requires the "
            "[bench] extra",
            reason="ranx_missing",
            hint="pip install -e '.[bench]'",
        ) from e
    return ranx


# ---- Per-config execution -------------------------------------------------


@dataclass
class ConfigResult:
    config: BenchConfig
    output_dir: Path
    status: str  # "ok" | "ingest_failed" | "query_failed"
    exit_code: int  # ingest exit code (0 if ok)
    chunks_generated: int
    duration_s: float
    metrics: dict[str, float]  # may be empty if status != "ok"
    run_dict: dict[str, dict[str, float]]  # qid -> chunk_id -> score (empty on failure)
    error: str = ""


def run_ingest_for_config(
    corpus_dir: Path, out_base: Path, cfg: BenchConfig
) -> tuple[int, int, str]:
    """`IngestRunner` を BenchConfig 付きで呼ぶ. (exit_code, chunks, err_msg) を返す."""
    ingest_cfg = IngestConfig(
        input_dir=corpus_dir,
        output_dir=out_base / cfg.name,
        skip_wiki=cfg.skip_wiki,
        target_chars=cfg.chunk_size,
        overlap_chars=cfg.overlap,
        llm_backend="anthropic",
        show_progress=False,
        # 単一 config で min_* を下げておくことで、tiny corpus でも NER 集計が
        # 通りやすい. bench は chunking params の差を見るのが主目的なので、
        # この既定値は妥当.
        min_mentions=1,
        min_chunks=1,
    )
    try:
        result = IngestRunner(ingest_cfg).run()
    except Exception as e:  # pragma: no cover - IngestRunner catches most
        return 10, 0, f"{type(e).__name__}: {e}"
    err_msg = ""
    if result.errors:
        first = result.errors[0]
        err_msg = f"{first.get('class', 'Error')}: {first.get('message', '')}"
    return result.exit_code, result.chunks_generated, err_msg


def run_queries(
    output_dir: Path, qrels: list[QrelEntry], *, top_k: int = 10
) -> dict[str, dict[str, float]]:
    """各 query を ingest 出力 dir に対して走らせ run dict を構築."""
    run_dict: dict[str, dict[str, float]] = {}
    for entry in qrels:
        qres = run_query_impl(output_dir, entry.query, top_k=top_k)
        hits_map: dict[str, float] = {}
        # exit_code != 0 (OOV など) は空 run として扱う — ranx は empty-run を
        # 無視するが、qid は残したいので 0-score を付与せず空 dict にする.
        if qres.exit_code == 0:
            for h in qres.hits:
                hits_map[h.chunk_id] = float(h.score)
        run_dict[entry.qid] = hits_map
    return run_dict


def evaluate_run(
    qrels: list[QrelEntry], run_dict: dict[str, dict[str, float]]
) -> dict[str, float]:
    """ranx で指定メトリクスを計算. 結果 dict は全 metric を含む (0.0 許容)."""
    ranx = _ensure_ranx()
    qrels_dict: dict[str, dict[str, int]] = {
        q.qid: dict(q.relevant) for q in qrels if q.relevant
    }
    # qid に relevant が 0 件のときは ranx が評価できないため除外する
    # (ranx < 0.4 は空 qrels dict を undefined behaviour で返す).
    if not qrels_dict:
        return {m: 0.0 for m in METRICS}
    run_filtered = {
        qid: (run_dict.get(qid) or {"__dummy__": 0.0}) for qid in qrels_dict
    }
    # ranx は run 側で「ヒット 0 件」を扱うために dummy key を入れると metric は 0
    # として評価される. ここでは relevant にない dummy なので recall/mrr/ndcg は
    # 期待通り 0 として集計される.
    qrels_obj = ranx.Qrels.from_dict(qrels_dict)
    run_obj = ranx.Run.from_dict(run_filtered)
    results = ranx.evaluate(qrels_obj, run_obj, list(METRICS))
    # evaluate は単一 metric 指定時に float を返す可能性があるが、list 渡しなので
    # dict が返るはず. 念のため dict 化.
    if isinstance(results, dict):
        return {m: float(results.get(m, 0.0)) for m in METRICS}
    # 単一 metric のスカラーで返ったとき (理論上 list 指定では起きない)
    return {m: float(results) for m in METRICS}


# ---- Table rendering ------------------------------------------------------


def _format_plain_table(rows: list[ConfigResult]) -> str:
    headers = [
        "config",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "chunks",
        "duration_s",
    ]
    lines: list[str] = []
    data: list[list[str]] = [headers]
    for r in rows:
        if r.status == "ok":
            data.append(
                [
                    r.config.name,
                    f"{r.metrics.get('recall@5', 0.0):.4f}",
                    f"{r.metrics.get('recall@10', 0.0):.4f}",
                    f"{r.metrics.get('mrr@10', 0.0):.4f}",
                    f"{r.metrics.get('ndcg@10', 0.0):.4f}",
                    str(r.chunks_generated),
                    f"{r.duration_s:.2f}",
                ]
            )
        else:
            data.append(
                [
                    r.config.name,
                    f"FAIL({r.status}:exit={r.exit_code})",
                    "-",
                    "-",
                    "-",
                    str(r.chunks_generated),
                    f"{r.duration_s:.2f}",
                ]
            )
    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]
    for ri, row in enumerate(data):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if ri == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _try_rich_table(rows: list[ConfigResult]) -> str | None:
    """rich が import できれば table 文字列を返す. 無ければ None."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        return None
    from io import StringIO

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    table = Table(title="RAG search benchmark")
    table.add_column("config")
    table.add_column("recall@5", justify="right")
    table.add_column("recall@10", justify="right")
    table.add_column("mrr@10", justify="right")
    table.add_column("ndcg@10", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("duration_s", justify="right")
    for r in rows:
        if r.status == "ok":
            table.add_row(
                r.config.name,
                f"{r.metrics.get('recall@5', 0.0):.4f}",
                f"{r.metrics.get('recall@10', 0.0):.4f}",
                f"{r.metrics.get('mrr@10', 0.0):.4f}",
                f"{r.metrics.get('ndcg@10', 0.0):.4f}",
                str(r.chunks_generated),
                f"{r.duration_s:.2f}",
            )
        else:
            table.add_row(
                r.config.name,
                f"FAIL({r.status}:exit={r.exit_code})",
                "-",
                "-",
                "-",
                str(r.chunks_generated),
                f"{r.duration_s:.2f}",
            )
    console.print(table)
    return buf.getvalue()


def render_table(rows: list[ConfigResult]) -> str:
    text = _try_rich_table(rows)
    if text is not None:
        return text
    return _format_plain_table(rows)


# ---- Main -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench.py",
        description=(
            "RAG 検索評価ベンチマーク (recall@k / MRR@10 / nDCG@10). "
            "本ベンチマークは自前 corpus のチューニング / regression 検出用途です. "
            "ツール間の quality 比較は対象外."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "corpus_dir",
        type=Path,
        help="入力コーパスディレクトリ (.txt を含む; IngestRunner に渡される)",
    )
    p.add_argument(
        "--configs",
        action="append",
        default=[],
        metavar="NAME=key:val,...",
        help=(
            "ベンチ対象 chunking config. 例: "
            "c256=chunk:256,overlap:32,wiki:off. "
            "キー chunk/overlap (int) と wiki (on|off). "
            "2 つ以上指定してください (comparison が目的のため)."
        ),
    )
    p.add_argument(
        "--qrels",
        type=Path,
        required=True,
        help="qrels JSONL (1 行 1 qid). {qid, query, relevant:[{chunk_id,grade}]}",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("bench_out"),
        help="各 config の ingest 出力を入れる親ディレクトリ (default: bench_out/)",
    )
    p.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="machine-readable report の出力パス (.json). 指定時のみ emit.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="query 1 件あたり取得する top-k (既定 10, recall@10 計算に必要)",
    )
    return p


def _config_error_to_exit(e: ConfigError) -> int:
    print(f"[error] {e}", file=sys.stderr)
    ctx = getattr(e, "context", {}) or {}
    hint = ctx.get("hint")
    if hint:
        print(f"        hint: {hint}", file=sys.stderr)
    return e.exit_code


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        # 1. Parse configs
        if not args.configs:
            raise ConfigError(
                "no --configs provided; at least 2 required for comparison",
                reason="insufficient_configs",
                count=0,
                hint="specify --configs NAME=chunk:N,overlap:N,wiki:on|off twice",
            )
        bench_configs: list[BenchConfig] = [parse_config_spec(s) for s in args.configs]
        if len(bench_configs) < 2:
            raise ConfigError(
                f"at least 2 --configs required, got {len(bench_configs)}",
                reason="insufficient_configs",
                count=len(bench_configs),
                hint="a single config is not a benchmark — specify another --configs",
            )
        # 2. Qrels
        qrels = load_qrels(args.qrels)
        # 3. ranx availability (fail-fast before running ingests)
        _ensure_ranx()
    except ConfigError as e:
        return _config_error_to_exit(e)

    # 4. corpus dir sanity (no files is handled by IngestRunner as ConfigError
    # exit 2; we let that propagate via the per-config result).
    if not args.corpus_dir.exists():
        e = ConfigError(
            f"corpus_dir does not exist: {args.corpus_dir}",
            reason="corpus_not_found",
            corpus_dir=str(args.corpus_dir),
        )
        return _config_error_to_exit(e)

    args.out.mkdir(parents=True, exist_ok=True)

    # 5. 各 config を順次実行
    results: list[ConfigResult] = []
    for cfg in bench_configs:
        out_dir = args.out / cfg.name
        t0 = time.perf_counter()
        exit_code, chunks, err = run_ingest_for_config(args.corpus_dir, args.out, cfg)
        if exit_code != 0:
            results.append(
                ConfigResult(
                    config=cfg,
                    output_dir=out_dir,
                    status="ingest_failed",
                    exit_code=exit_code,
                    chunks_generated=chunks,
                    duration_s=time.perf_counter() - t0,
                    metrics={m: 0.0 for m in METRICS},
                    run_dict={},
                    error=err,
                )
            )
            print(
                f"[warn] config {cfg.name}: ingest failed (exit={exit_code}): {err}",
                file=sys.stderr,
            )
            continue
        try:
            run_dict = run_queries(out_dir, qrels, top_k=args.top_k)
            metrics = evaluate_run(qrels, run_dict)
        except ConfigError as e:
            return _config_error_to_exit(e)
        except Exception as e:  # pragma: no cover - defensive
            results.append(
                ConfigResult(
                    config=cfg,
                    output_dir=out_dir,
                    status="query_failed",
                    exit_code=0,
                    chunks_generated=chunks,
                    duration_s=time.perf_counter() - t0,
                    metrics={m: 0.0 for m in METRICS},
                    run_dict={},
                    error=f"{type(e).__name__}: {e}",
                )
            )
            continue
        results.append(
            ConfigResult(
                config=cfg,
                output_dir=out_dir,
                status="ok",
                exit_code=0,
                chunks_generated=chunks,
                duration_s=time.perf_counter() - t0,
                metrics=metrics,
                run_dict=run_dict,
            )
        )

    # 6. テーブル出力
    table_text = render_table(results)
    print(table_text)

    # 7. JSON 出力 (optional)
    if args.json_path is not None:
        report: dict[str, Any] = {
            "schema_version": 1,
            "corpus_dir": str(args.corpus_dir),
            "qrels_path": str(args.qrels),
            "qid_count": len(qrels),
            "metrics": list(METRICS),
            "configs": [
                {
                    "name": r.config.name,
                    "chunk_size": r.config.chunk_size,
                    "overlap": r.config.overlap,
                    "skip_wiki": r.config.skip_wiki,
                    "status": r.status,
                    "exit_code": r.exit_code,
                    "chunks_generated": r.chunks_generated,
                    "duration_s": r.duration_s,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in results
            ],
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        with args.json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # 8. exit code: 全 config が ok なら 0, いずれか ingest_failed なら最初の
    # exit_code を propagate (但し少なくとも 1 件成功していれば exit 0 とする —
    # bench は partial 成功でも比較 table は有用なため).
    if any(r.status == "ok" for r in results):
        return 0
    # 全滅時は最初の失敗 exit_code.
    for r in results:
        if r.status != "ok":
            return r.exit_code
    return 0


def main() -> int:  # pragma: no cover - thin wrapper
    return run()


if __name__ == "__main__":
    sys.exit(main())
