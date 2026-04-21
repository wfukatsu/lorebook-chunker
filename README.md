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

```bash
pip install -e .[dev]
python -m spacy validate  # ja_ginza_electra の導入確認
```

`ja_ginza_electra` は PyPI 未公開のため URL 依存で取得されます。インストールに失敗する場合は [Ginza リリースページ](https://github.com/megagonlabs/ginza/releases) で最新の wheel URL を確認してください。

LLM バックエンドの準備:
- **Anthropic** (既定): `export ANTHROPIC_API_KEY=...`
- **Ollama** (オフライン): `ollama pull qwen2.5:7b-instruct` 等

## 使い方 (最短経路)

```bash
# 処理
chunking ingest samples/ out/

# 検索 (ベースライン)
chunking query out/ "合併の背景" --top-k 5

# 健全性チェック
chunking lint out/
```

## 既知の限界

- **TF-IDF 単独の言い換えクエリ脆弱性**: 同義語や表現違いのクエリ (例: 「合併の背景」→「経営統合の経緯」) では関連チャンクを取りこぼすことがあります。本番 RAG 検索は dense retrieval (埋め込みモデル) に委ねる前提です。
- **固有名詞の表記ゆれ未統合**: 「スカラー」「Scalar」「スカラ商事」は NER が別エンティティとして扱い、別 wiki ページになります。`lint` で表記近似候補を列挙しますが、統合は手動判断です。
- **TF-IDF IDF 安定性**: 数十チャンク規模のコーパスでは IDF 統計が不安定になります (文献通り)。本番コーパス規模で運用してください。
- **Ollama 日本語要約品質**: モデル依存。本番品質は Anthropic を推奨。
- **単一プロセス前提**: `log.md` に同時書き込みする複数 ingest 実行はサポートしません。

## ライセンス

MIT
