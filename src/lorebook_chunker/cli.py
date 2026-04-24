"""CLI entrypoint: `lorebook-chunker ingest | query | lint`."""
from __future__ import annotations

import argparse
import codecs
import sys
from typing import Sequence

from .errors import ConfigError, describe_exit_codes

IDENTITY_BANNER = (
    "lorebook-chunker: 日本語 RAG 前処理・コーパス健全性・一級エンティティ知識ベース生成ツール "
    "(本番検索は下流 vector store で行う前提)"
)

# ingest は LorebookError 階層を通じて exit code を決定するため、
# epilog は errors.describe_exit_codes() から生成 (single source of truth).
INGEST_EPILOG = describe_exit_codes()

# NOTE: lint / query の exit code は LorebookError 階層とは別 namespace の
# subcommand-local semantics:
#   lint  — 0/1/2 (clean / warnings / fatal) の severity 集計
#   query — 0/2/3/4 (hit / missing / corrupted / no-match)  の検索結果
# ingest の LorebookError exit code (0, 2-6, 10-17) と衝突しないが、
# 意味も異なるので errors.py に混ぜず subcommand ごとにハードコードする.
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
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="入力ディレクトリをサブディレクトリまで再帰的に探索する (既定: トップレベルのみ)",
    )
    p.add_argument(
        "--glob",
        dest="globs",
        default="*.txt",
        metavar="PATTERN",
        help=(
            "入力ファイルを絞り込む glob パターン. カンマ区切りで複数指定可 "
            "(例: '*.txt,*.md'). 既定: '*.txt'. 絶対パスや空文字は不可."
        ),
    )
    p.add_argument(
        "--encoding",
        default="auto",
        metavar="NAME",
        help=(
            "入力ファイルのエンコーディング. 'auto' (既定) は utf-8-sig を試し、"
            "失敗時は charset-normalizer ([full] extra 導入時のみ) で検出. "
            "明示指定は strict decode: 'utf-8' / 'utf-8-sig' / 'cp932' / "
            "'shift_jis' / 'euc_jp' など Python codec 名."
        ),
    )
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
        "--llm-parallelism",
        type=int,
        default=None,
        metavar="N",
        help=(
            "wiki 生成時の LLM 同時呼出数 (既定: anthropic=5, ollama=3). "
            "Anthropic は公式 concurrency 制限内で 10 まで上げられる. "
            "Ollama は OLLAMA_NUM_PARALLEL と揃える (7B モデル + 32GB なら 3 が目安). "
            "1 を指定すると旧来のシリアル挙動になる."
        ),
    )
    p.add_argument(
        "--analyzer-backend",
        choices=["electra", "ginza"],
        default="electra",
        help=(
            "日本語 NLP バックエンド. 既定 electra (ja_ginza_electra, transformer). "
            "ginza は transformer を持たない軽量モデル (ja_ginza) で CPU 推論が大幅に速い "
            "(NER 粒度が若干違うので精度トレード). `pip install -e '.[fast]'` が必要."
        ),
    )
    p.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help=(
            "transformer 推論デバイス. 既定 cpu. mps は Apple Silicon の Metal で "
            "ELECTRA を高速化 (--analyzer-backend electra のみ効果). "
            "MPS 時は n_process が自動的に 1 に強制される (MPS コンテキストは "
            "プロセス間共有できないため). 数値は CPU と bit-exact ではないので "
            "POS/NER の境界で vocab/chunks が僅かに変わる可能性あり."
        ),
    )
    p.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="出力形式 (既定: human). json 指定時は summary が stdout に出る",
    )
    p.add_argument("--quiet", action="store_true", help="identity banner と human success 行を抑止 (進捗も抑止)")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="stderr への進捗表示を抑止 (--quiet でも自動で off)",
    )
    p.add_argument(
        "--verify-swap",
        action="store_true",
        help=(
            "出力ディレクトリ切替後に staging/target の SHA-256 照合を実施する (opt-in). "
            "同一 FS 上では os.replace が inode 操作のため tautological; "
            "主に cross-device (EXDEV) 経路での破損検知に意味を持つ. "
            "数秒〜数十秒のオーバーヘッド."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "事前検証のみ実行: doctor サブコマンドの環境チェック + 入力ファイル探索 + "
            "先頭ファイルの encoding probe (64 KiB sample). "
            "出力ディレクトリは作成せず、stdout に JSON サマリを出力. "
            "doctor が critical failure を返した場合はその exit 18 を propagate する."
        ),
    )


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
    # doctor は ingest 系とは別 exit code 空間 (0/1/18) を使うため、
    # subparser 登録のみ cli.py 側で行い、引数定義は doctor.py に委譲する.
    from lorebook_chunker.doctor import add_doctor_subparser
    add_doctor_subparser(subparsers)
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
    if args.command == "doctor":
        from lorebook_chunker.doctor import run_doctor
        return run_doctor(args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
