"""CLI entrypoint: `lorebook-chunker ingest | query | lint`."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

IDENTITY_BANNER = (
    "lorebook-chunker: 日本語 RAG 前処理・コーパス健全性・一級エンティティ知識ベース生成ツール "
    "(本番検索は下流 vector store で行う前提)"
)

INGEST_EPILOG = """\
exit codes:
  0   success
  2   no input .txt files under input_dir
  3   LLM backend unavailable (permanent error at init)
  4   analyzer init failed
  5   chunking produced zero chunks
  6   wiki generation aborted (systemic failure detected)
  10  unexpected error (see log / stderr)
"""

LINT_EPILOG = """\
exit codes:
  0  no fatal and no warning findings
  1  warning findings only (no fatal)
  2  at least one fatal finding
"""

QUERY_EPILOG = """\
exit codes:
  0  at least one matching chunk returned
  2  required artifacts missing under output_dir
  3  artifacts corrupted or schema mismatch
  4  query produced zero tokens or zero hits (OOV / no-match)
"""


def _add_ingest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ingest",
        help="入力 .txt をチャンク化・TF-IDF・エンティティ wiki に変換",
        description=IDENTITY_BANNER,
        epilog=INGEST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_dir", help="入力ディレクトリ (.txt を含む)")
    p.add_argument("output_dir", help="出力ディレクトリ (破壊的全再生成)")
    p.add_argument("--skip-wiki", action="store_true", help="エンティティ wiki 生成をスキップ")
    p.add_argument("--max-llm-calls", type=int, default=None, help="LLM 呼び出しの上限 (entity 試行のみ, pre-flight は除外)")
    p.add_argument("--force-regenerate", action="store_true", help="全 wiki を無条件再生成")
    p.add_argument("--retry-failed", action="store_true", help="前回 failed/budget_skipped を再試行")
    p.add_argument(
        "--llm-backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM バックエンド (既定: anthropic)",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        metavar="MODEL",
        help=(
            "選択したバックエンドに渡すモデル名 (既定: バックエンド既定値). "
            "例: --llm-backend ollama --llm-model qwen3:8b / "
            "--llm-backend anthropic --llm-model claude-haiku-4-5"
        ),
    )
    p.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="出力形式 (既定: human). json 指定時は summary が stdout に出る",
    )
    p.add_argument("--quiet", action="store_true", help="identity banner と human success 行を抑止")


def _add_query(subparsers: argparse._SubParsersAction) -> None:
    from lorebook_chunker.query import positive_int

    p = subparsers.add_parser(
        "query",
        help="TF-IDF コサイン類似度で top-K チャンクを返す (ベースライン検索)",
        description=IDENTITY_BANNER,
        epilog=QUERY_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dir", help="ingest が生成した出力ディレクトリ")
    p.add_argument("query_text", help="検索文")
    p.add_argument(
        "--top-k",
        type=positive_int,
        default=5,
        help="返すチャンク数 (正の整数, 既定 5)",
    )
    p.add_argument(
        "--format",
        choices=["human", "json"],
        default=None,
        help="出力形式. 未指定時 stdout が tty でなければ json",
    )


def _add_lint(subparsers: argparse._SubParsersAction) -> None:
    from lorebook_chunker.lint import (
        DEFAULT_DEGENERATE_L2,
        DEFAULT_DEGENERATE_NNZ,
        DEFAULT_DUPLICATE_COSINE,
        DEFAULT_LEVENSHTEIN_RATIO,
        DEFAULT_LINT_PAIRWISE_THRESHOLD,
    )

    p = subparsers.add_parser(
        "lint",
        help="コーパス健全性を検査 (致命/警告/情報の3段階)",
        description=IDENTITY_BANNER,
        epilog=LINT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dir", help="ingest が生成した出力ディレクトリ")
    p.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="出力形式 (既定: human). json 指定時は lint.json も書き出す",
    )
    # F-060: lint の tuning thresholds を CLI に露出
    p.add_argument(
        "--pairwise-threshold",
        type=int,
        default=DEFAULT_LINT_PAIRWISE_THRESHOLD,
        help="pairwise 比較を打ち切る N の閾値",
    )
    p.add_argument(
        "--duplicate-cosine",
        type=float,
        default=DEFAULT_DUPLICATE_COSINE,
        help="重複と判定するコサイン類似度の閾値",
    )
    p.add_argument(
        "--degenerate-l2",
        type=float,
        default=DEFAULT_DEGENERATE_L2,
        help="縮退とみなす L2 ノルムの下限",
    )
    p.add_argument(
        "--degenerate-nnz",
        type=int,
        default=DEFAULT_DEGENERATE_NNZ,
        help="縮退とみなす nnz の下限",
    )
    p.add_argument(
        "--levenshtein-ratio",
        type=float,
        default=DEFAULT_LEVENSHTEIN_RATIO,
        help="表記近似と判定する SequenceMatcher 比率",
    )
    p.add_argument(
        "--target-chunk-chars",
        type=int,
        default=500,
        help="target_chunk_chars (ingest 側の target_chars と揃える)",
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1500,
        help="max_chunk_chars (ingest 側と揃える)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lorebook-chunker",
        description=IDENTITY_BANNER,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest(subparsers)
    _add_query(subparsers)
    _add_lint(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        from lorebook_chunker.ingest import run_ingest
        return run_ingest(args)
    if args.command == "query":
        from lorebook_chunker.query import run_query
        return run_query(args)
    if args.command == "lint":
        from lorebook_chunker.lint import run_lint
        return run_lint(args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
