#!/usr/bin/env bash
# fast-ingest.sh — ハードウェアと LLM バックエンドに応じた最速設定で `lorebook-chunker ingest` を起動する.
#
#   ./scripts/fast-ingest.sh <input_dir> <output_dir> [追加 CLI 引数 ...]
#
# 最速化の内訳:
#   1. ELECTRA の単一パス化 (analyze_documents) は既定で有効 (コード側で自動).
#   2. Apple Silicon を検知したら --device mps で ELECTRA を Metal に流す (実測 2.9〜4.7x).
#   3. nlp.pipe の batch_size / n_process を環境に合わせて調整:
#        - CPU: LOREBOOK_CHUNKER_N_PROCESS=6 (README の 4 より気持ち上). 8 以降は IPC cost で頭打ち.
#        - GPU (mps/cuda): ingest 側で n_process=1 に強制. batch_size を 64 に寄せて GPU 占有率を上げる.
#   4. wiki 生成の LLM 並列度:
#        - Anthropic: 10 (公式 concurrency 上限内の最大).
#        - Ollama:    3 (典型 M2 Pro 32GB / qwen3:8b の安全点. OLLAMA_NUM_PARALLEL と揃える).
#   5. rapidfuzz が入っていれば lint の N² が 10-50x 速くなる (本スクリプトでは install しないが通知).
#
# 環境変数による上書き:
#   LLM_BACKEND=anthropic|ollama   (既定 anthropic)
#   LLM_MODEL=<model-id>           (既定 backend ごとの haiku-4-5 / qwen3:8b)
#   LLM_PARALLELISM=<int>          (既定 backend ごとの 10 / 3)
#   DEVICE=cpu|mps|cuda            (既定: uname で自動判定)
#   BATCH_SIZE=<int>               (既定: GPU なら 64, CPU なら 32)
#   N_PROCESS=<int>                (既定: CPU は 6, GPU は 1 強制)
#
# 依存:
#   - `lorebook-chunker` コマンドが PATH にある (pip install -e '.[dev]' 済みの venv を activate 推奨).
#   - Anthropic 利用時は ANTHROPIC_API_KEY が export 済み.
#   - Ollama 利用時は `ollama serve` が起動済み + 指定モデルが pull 済み.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
usage: fast-ingest.sh <input_dir> <output_dir> [追加 CLI 引数 ...]

例:
  # Apple Silicon + Anthropic (既定) — 最速
  ANTHROPIC_API_KEY=sk-... ./scripts/fast-ingest.sh samples/ out/

  # Ollama オフライン + Anthropic より高速設定
  LLM_BACKEND=ollama LLM_MODEL=qwen3:8b ./scripts/fast-ingest.sh samples/ out/

  # 既存 out/ のキャッシュを使って失敗分だけ retry
  ./scripts/fast-ingest.sh samples/ out/ --retry-failed

  # wiki を切って chunks + vocab だけ (開発ループ最速)
  ./scripts/fast-ingest.sh samples/ out/ --skip-wiki
USAGE
  exit 64
fi

input_dir=$1
output_dir=$2
shift 2

# --- device 自動判定 ------------------------------------------------------
if [[ -z "${DEVICE:-}" ]]; then
  case "$(uname -sm)" in
    "Darwin arm64") DEVICE=mps ;;   # Apple Silicon
    *)              DEVICE=cpu ;;
  esac
fi

# --- LLM バックエンド既定 -------------------------------------------------
LLM_BACKEND="${LLM_BACKEND:-anthropic}"
case "$LLM_BACKEND" in
  anthropic)
    LLM_MODEL="${LLM_MODEL:-claude-haiku-4-5}"
    LLM_PARALLELISM="${LLM_PARALLELISM:-10}"
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "[fast-ingest] ANTHROPIC_API_KEY が未設定です. export してから再実行してください." >&2
      exit 3
    fi
    ;;
  ollama)
    LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
    LLM_PARALLELISM="${LLM_PARALLELISM:-3}"
    # Ollama 本体も同じ parallelism で受け付ける必要あり. すでに serve 中なら env 反映は next restart.
    : "${OLLAMA_NUM_PARALLEL:=${LLM_PARALLELISM}}"
    export OLLAMA_NUM_PARALLEL
    ;;
  *)
    echo "[fast-ingest] 未対応バックエンド: $LLM_BACKEND (anthropic|ollama)" >&2
    exit 64
    ;;
esac

# --- nlp.pipe チューニング ------------------------------------------------
# GPU 時は ingest.py 側で n_process=1 に強制される (GPU コンテキストはプロセス間共有不可).
# そこでは batch_size を積んで GPU バッチ推論の利点を引き出す.
if [[ "$DEVICE" == "cpu" ]]; then
  default_batch=32
  default_nproc=6
else
  default_batch=64
  default_nproc=1
fi
export LOREBOOK_CHUNKER_BATCH_SIZE="${BATCH_SIZE:-$default_batch}"
export LOREBOOK_CHUNKER_N_PROCESS="${N_PROCESS:-$default_nproc}"

# --- rapidfuzz の有無 (lint 高速化) --------------------------------------
if ! python -c "import rapidfuzz" >/dev/null 2>&1; then
  echo "[fast-ingest] ヒント: \`pip install rapidfuzz\` を入れると lint の N² 類似度検出が 10-50x 速くなります." >&2
fi

echo "[fast-ingest] device=$DEVICE backend=$LLM_BACKEND model=$LLM_MODEL" \
     "parallelism=$LLM_PARALLELISM batch=$LOREBOOK_CHUNKER_BATCH_SIZE" \
     "n_process=$LOREBOOK_CHUNKER_N_PROCESS" >&2

exec lorebook-chunker ingest "$input_dir" "$output_dir" \
  --analyzer-backend electra \
  --device "$DEVICE" \
  --llm-backend "$LLM_BACKEND" \
  --llm-model "$LLM_MODEL" \
  --llm-parallelism "$LLM_PARALLELISM" \
  "$@"
