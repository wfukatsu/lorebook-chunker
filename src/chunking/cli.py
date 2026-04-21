"""CLI entrypoint: `chunking ingest | query | lint`."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

IDENTITY_BANNER = (
    "chunking: 日本語 RAG 前処理・コーパス健全性・一級エンティティ知識ベース生成ツール "
    "(本番検索は下流 vector store で行う前提)"
)


def _add_ingest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ingest",
        help="入力 .txt をチャンク化・TF-IDF・エンティティ wiki に変換",
        description=IDENTITY_BANNER,
    )
    p.add_argument("input_dir", help="入力ディレクトリ (.txt を含む)")
    p.add_argument("output_dir", help="出力ディレクトリ (破壊的全再生成)")
    p.add_argument("--skip-wiki", action="store_true", help="エンティティ wiki 生成をスキップ")
    p.add_argument("--max-llm-calls", type=int, default=None, help="LLM 呼び出しの上限")
    p.add_argument("--force-regenerate", action="store_true", help="全 wiki を無条件再生成")
    p.add_argument("--retry-failed", action="store_true", help="前回 failed/budget_skipped を再試行")
    p.add_argument(
        "--llm-backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM バックエンド (既定: anthropic)",
    )
    p.add_argument("--config", default=None, help="設定ファイル (optional)")


def _add_query(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "query",
        help="TF-IDF コサイン類似度で top-K チャンクを返す (ベースライン検索)",
        description=IDENTITY_BANNER,
    )
    p.add_argument("output_dir", help="ingest が生成した出力ディレクトリ")
    p.add_argument("query_text", help="検索文")
    p.add_argument("--top-k", type=int, default=5, help="返すチャンク数 (既定 5)")


def _add_lint(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "lint",
        help="コーパス健全性を検査 (致命/警告/情報の3段階)",
        description=IDENTITY_BANNER,
    )
    p.add_argument("output_dir", help="ingest が生成した出力ディレクトリ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunking",
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
        from chunking.ingest import run_ingest
        return run_ingest(args)
    if args.command == "query":
        from chunking.query import run_query
        return run_query(args)
    if args.command == "lint":
        from chunking.lint import run_lint
        return run_lint(args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
