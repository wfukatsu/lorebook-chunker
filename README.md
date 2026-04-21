# chunking — 日本語 RAG 向けチャンク化 + TF-IDF + 固有名詞 Wiki CLI

> **本スクリプトのアイデンティティ**
> RAG 用コーパスの前処理・健全性・一級エンティティ知識ベース生成ツールです。
> `query` サブコマンドはコーパス整形時の確認用 (TF-IDF コサイン類似度ベースのベースライン) であって、本番運用の検索ではありません。本番検索は下流の vector store / 検索基盤で行う前提です。

## できること

3 つのサブコマンドを提供します:

- **`ingest`**: `.txt` ファイル群 → チャンク化 + TF-IDF 疎ベクトル + キーワード + 固有名詞 wiki (LLM 要約) を一括生成
- **`query`**: 保存済み TF-IDF 語彙でクエリ文をベクトル化し、コサイン類似度で上位 K チャンクを返す (ベースライン検索、本番用途ではない)
- **`lint`**: コーパスの健全性 (空/重複/縮退チャンク、孤立 wiki、表記近似エンティティ) をチェック

## 出力契約

2 系統:

1. **RAG 検索器入力**: `chunks.jsonl` + `vocab.npz` + `analyzer.json`
2. **エンティティ知識ベース (一級出力)**: `entities/*.md` + `entities/manifest.json` + `index.md`

## インストール

### 前提環境

- **Python 3.11 または 3.12**. 3.13 以降は `ja-ginza-electra` 依存の `tokenizers<0.14` に prebuilt wheel が無く、Rust ソースビルドも失敗するため `pyproject.toml` で `<3.13` に固定しています。macOS では `brew install python@3.11` で導入できます。
- macOS / Linux (M1/M2/Intel). Windows は ChunkRecord のパス正規化 (`Path(...).relative_to(...).as_posix()`) では対応していますが、e2e は未検証です。

### 依存インストール

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m spacy validate  # ja_ginza_electra の導入確認 (✔ が出れば OK)
```

`ja_ginza_electra` は PyPI 未公開のため GitHub Releases の wheel を URL 依存 (sha256 ピン) で取得します。初回 `chunking ingest` 実行時に ELECTRA transformer 本体 (~400MB) が HuggingFace Hub から追加ダウンロードされます (オフラインなら `HF_HUB_OFFLINE=1` + 事前キャッシュが必要)。

### LLM バックエンドの準備

- **Anthropic** (既定): `export ANTHROPIC_API_KEY=...`。既定モデルは `claude-haiku-4-5`。
- **Ollama** (ローカル / オフライン): Ollama 本体をインストールし、モデルを pull。
  ```bash
  brew install ollama        # macOS; or see https://ollama.com/download
  ollama serve &             # 常駐デーモン (既に起動済みならスキップ)
  ollama pull qwen3:8b       # ~5.2GB. qwen2.5:7b-instruct-q4_K_M でも可.
  ```
  `qwen3:*` 系は `think` チャネルに全トークンを消費するため、本 CLI は内部で `think=False` を渡して最終出力のみを取得します。

## 使い方 (最短経路)

```bash
# 既定 (Anthropic, claude-haiku-4-5) で処理
chunking ingest samples/ out/

# Ollama ローカルモデルを明示指定
chunking ingest samples/ out/ --llm-backend ollama --llm-model qwen3:8b

# Anthropic で別モデルを使う
chunking ingest samples/ out/ --llm-backend anthropic --llm-model claude-sonnet-4-5

# 検索 (ベースライン)
chunking query out/ "合併の背景" --top-k 5

# 健全性チェック
chunking lint out/
```

### `--llm-model` について

| バックエンド | `--llm-model` 未指定時の既定 | 例 |
|---|---|---|
| `anthropic` | `claude-haiku-4-5` | `--llm-model claude-sonnet-4-5` |
| `ollama`    | `qwen2.5:7b-instruct-q4_K_M` (Ollama 側で pull 済みが必要) | `--llm-model qwen3:8b` |

`--llm-model` は `--llm-backend` で選んだバックエンドに透過的に渡されます。バックエンドごとに独立したフラグを分ける代わりに、「バックエンド × モデル名」の 1 ペアで指定できる設計です。

## 終了コード

### `ingest`

| code | 意味 |
|---:|---|
| 0  | 成功 |
| 2  | `input_dir` に `.txt` が見つからない |
| 3  | LLM バックエンドが init 時点で恒久エラー |
| 4  | analyzer 初期化失敗 |
| 5  | チャンク 0 件 (全ファイル空 / decode 失敗等) |
| 6  | wiki 生成が systemic failure で abort |
| 10 | 予期せぬ例外 (stderr / log を参照) |

### `query`

| code | 意味 |
|---:|---|
| 0  | 1 件以上ヒット |
| 2  | 必要成果物 (`chunks.jsonl`/`vocab.npz`/`analyzer.json`) が揃っていない |
| 3  | 成果物が破損 / row 数不整合 |
| 4  | クエリが OOV / ゼロトークン / ゼロヒット |

### `lint`

| code | 意味 |
|---:|---|
| 0  | 致命も警告もゼロ |
| 1  | 警告のみ (致命ゼロ) |
| 2  | 致命が 1 件以上 |

## 既知の限界

- **TF-IDF 単独の言い換えクエリ脆弱性**: 同義語や表現違いのクエリ (例: 「合併の背景」→「経営統合の経緯」) では関連チャンクを取りこぼすことがあります。本番 RAG 検索は dense retrieval (埋め込みモデル) に委ねる前提です。
- **固有名詞の表記ゆれ未統合**: 「スカラー」「Scalar」「スカラ商事」は NER が別エンティティとして扱い、別 wiki ページになります。`lint` で表記近似候補を列挙しますが、統合は手動判断です。
- **TF-IDF IDF 安定性**: 数十チャンク規模のコーパスでは IDF 統計が不安定になります (文献通り)。本番コーパス規模で運用してください。
- **Ollama 日本語要約品質**: モデル依存。本番品質は Anthropic を推奨。
- **単一プロセス前提**: `log.md` に同時書き込みする複数 ingest 実行はサポートしません。
- **Ginza 5.2 × spacy 3.8 互換シム**: `ja-ginza-electra 5.2.0` 本体は spacy 3.5 前後を想定した古い設定で配布されています。`analyzer.py` 側で (a) `compound_splitter.split_mode` の Config 上書きと、(b) `token._.ne` 拡張属性を `ginza.ENE_ONTONOTES_MAPPING` から派生する `spacy.tokens.Token.set_extension` 登録、を実行時に行うことで spacy 3.7〜3.8 でも動作します。Ginza 本体が更新された場合は shim 削除を検討してください。

## ライセンス

MIT
