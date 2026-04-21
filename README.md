# chunking — 日本語 RAG 向けチャンク化 + TF-IDF + 固有名詞 Wiki CLI

> **本スクリプトのアイデンティティ**
> RAG 用コーパスの前処理・健全性・一級エンティティ知識ベース生成ツールです。
> `query` サブコマンドはコーパス整形時の確認用 (TF-IDF コサイン類似度ベースのベースライン) であって、本番運用の検索ではありません。本番検索は下流の vector store / 検索基盤で行う前提です。

## できること

3 つのサブコマンドを提供します:

- **`ingest`**: `.txt` ファイル群 → チャンク化 + TF-IDF 疎ベクトル + キーワード + 固有名詞 wiki (LLM 要約) を一括生成
- **`query`**: 保存済み TF-IDF 語彙でクエリ文をベクトル化し、コサイン類似度で上位 K チャンクを返す (ベースライン検索、本番用途ではない)
- **`lint`**: コーパスの健全性 (空/重複/縮退チャンク、孤立 wiki、表記近似エンティティ) をチェック

---

## アーキテクチャ

### データフロー (ingest)

```
 input_dir/*.txt
      │
      ▼
 ┌──────────────────┐
 │ normalize_text   │  NFKC + LF-only + 末尾 strip + 連続空白 collapse + 全文 strip
 └──────────────────┘
      │                                                               (正規化後テキスト)
      ▼
 ┌──────────────────┐   Ginza ja_ginza_electra
 │ JapaneseAnalyzer │   ├─ iter_sentences         (spaCy sents)
 │ (spacy.load)     │   ├─ tokenize_for_tfidf     (Sudachi lemma, POS allowlist)
 └──────────────────┘   └─ iter_entities          (BIO walk over doc.ents)
      │                                                       │
      ├─ 文境界 + 文字数でチャンク生成                        │ 各チャンク text
      ▼                                                       ▼
 ┌─────────────┐        ┌─────────────────────┐    ┌──────────────────┐
 │ JpChunker   │───▶───▶│ TfidfBuilder        │    │ aggregate_entities│
 │ target/over │        │ (TfidfVectorizer,   │    │ min_mentions / chu│
 │ lap/soft-   │ chunk  │  custom analyzer)   │    │ nks でフィルタ + │
 │ split       │ 列     │                     │    │ OntoNotes5 label  │
 └─────────────┘        └─────────────────────┘    └──────────────────┘
      │                         │                          │
      │   row_index はコーパス   │ top_keywords/chunk       │ EntityAggregate[]
      │   全体で global 採番     │ + vocab_terms / idf     │
      ▼                         ▼                          ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                           staging/ (*.staging)                    │
 │  chunks.jsonl  vocab.npz  analyzer.json                          │
 │  entities/<LABEL>__<name>_<sha6>.md  manifest.json  index.md     │
 │  log.md  ingest_result.json                                      │
 └──────────────────────────────────────────────────────────────────┘
      │                                  ▲
      │ wiki generation                  │ 既存 output の manifest.json を pre-populate
      ▼                                  │ (source_hash 一致なら cache skip)
 ┌────────────────────────────┐          │
 │ WikiGenerator              │          │
 │  - pre-flight 1 call       │──────────┘
 │  - mention_count DESC ソート
 │  - 決定論順序で LLM 呼出   │       Anthropic / Ollama (LLMClient 実装を切替)
 │  - 3 回まで retry          │
 │  - budget / systemic abort │
 │  - source_hash でキャッシュ │
 └────────────────────────────┘
      │
      ▼ atomic swap (os.rename + cross-fs fallback + backup)
 ┌────────────────────────────┐
 │  output_dir/   (確定版)    │
 └────────────────────────────┘
```

`query` と `lint` は既存 `output_dir/` を読むだけで、**input_dir は参照しません**。`analyzer.json` の `strict_match` が現ランタイムと一致することを `JapaneseAnalyzer.load_and_verify` で検証し、不一致なら `AnalyzerVersionMismatchError`。

### パッケージ構成

```
src/chunking/
  __main__.py           python -m chunking → cli.main()
  cli.py                argparse / サブコマンド登録 / identity banner
  normalize.py          NFKC + LF + trim + collapse を 1 関数で (決定論契約)
  analyzer.py           Ginza + Sudachi のラッパ。Ginza 5.2 ↔ spacy 3.8 の shim 内蔵
  chunker.py            文境界を跨いだ char-target + overlap + soft-split
  tfidf.py              TfidfBuilder: vocab.npz (matrix + vocab + idf + config) を単一 .npz に
  ner.py                aggregate_entities: mention/chunk しきい値で刈り、決定論順序に整列
  schema.py             ChunkRecord / AnalyzerConfig / EntityAggregate / Error 型
  wiki.py               pre-flight + retry + budget + systemic-abort + source_hash cache
  lint.py               重複/縮退/孤立 wiki/表記近似 の 3 段階 (致命/警告/情報) 報告
  ingest.py             IngestRunner: 全 Unit を結線 + staging dir atomic swap
  query.py              cosine top-K + OOV/zero-hit ハンドリング
  _io.py                load_chunks (query/lint 共通)
  llm/
    __init__.py         LLMClient Protocol + GenerateResult + 典型 Error 階層 + get_client()
    anthropic_client.py AnthropicLLMClient (Claude Messages API)
    ollama_client.py    OllamaLLMClient (ローカル, qwen3 用の think=False シム入り)
    openai_client.py    OpenAIPlaceholderClient (init 時点で LLMPermanentError)
  resources/
    preflight_prompt.txt 起動時に 1 回だけ投げる pre-flight プロンプト
```

### 出力契約 (2 系統)

`ingest` の出力は 2 つの独立した契約で構成されます。下流ツールはどちらか一方だけを消費しても成立します。

#### (a) RAG 検索器入力

下流の vector store / 検索基盤に渡す想定。

```
output_dir/
├── chunks.jsonl      # 1 行 = 1 チャンク (ChunkRecord)
├── vocab.npz         # TF-IDF 行列 + 語彙 + IDF + vectorizer 設定
└── analyzer.json     # ingest 時の解析器設定 (strict / compat)
```

**`chunks.jsonl`** の 1 行 (`ChunkRecord`):

```json
{
  "chunk_id": "6cac241cf928",
  "row_index": 0,
  "source": "01_news.txt",
  "char_start": 0,
  "char_end": 285,
  "text": "スカラー商事は本日、...",
  "top_keywords": [{"term": "合併", "tfidf": 0.274}, ...],
  "entities": [{"name": "スカラー商事", "ner_label": "ORG",
                "char_start": 0, "char_end": 6}, ...]
}
```

- `chunk_id` は `sha256(posix_source + char_start + char_end + text)` の先頭 12 桁。同一入力なら R8b (破壊的全再生成) でも同一。
- `row_index` は **コーパス全体で global に採番** (`vocab.npz` の matrix row と 1:1)。ファイルをまたがる並びで 0 始まり連番。
- `char_start` / `char_end` は **正規化後テキストでの char offset**。元テキスト (入力ファイル) と往復することは前提にしていません。
- `sparse_vec` は **インラインでは持ちません** (`vocab.npz` の `matrix_*` 配列と `row_index` で参照)。

**`vocab.npz`** (numpy savez_compressed で `allow_pickle=False` 保存):

```
matrix_data / matrix_indices / matrix_indptr / matrix_shape   # scipy CSR 分解
vocabulary_terms                                               # np.str_ 配列
idf                                                            # float64 配列
config_json                                                    # utf-8 bytes (JSON)
```

`query` 側は `TfidfVectorizer` を pickle 復元せず、この 4 点から `vocabulary_` と `idf_` を埋めて現在の `JapaneseAnalyzer.tokenize_for_tfidf` を callable analyzer として注入し直します。

**`analyzer.json`**:

```json
{
  "version": 1,
  "strict_match": {
    "model_name": "ja_ginza_electra",
    "model_checksum": "sha256:...",
    "split_mode": "C",
    "pos_allowlist": ["NOUN", "VERB", "ADJ", "PROPN"],
    "stopwords": [...],
    "lemma_rules": "lemma_ field as-is",
    "sudachidict_binary_sha256": "sha256:...",
    "normalization": {"nfkc": true, "lf_only": true,
                       "strip_trailing": true, "collapse_spaces": true}
  },
  "compat_match": {
    "ginza": "5.2.0", "spacy": "3.8.14", "sudachipy": "0.6.11",
    "sudachidict_package": "sudachidict_core",
    "sudachidict_package_version": "20260116"
  },
  "tfidf": {"min_df": 1, "max_df": 0.95, ...}
}
```

load 検証:
- `strict_match.*` が 1 つでも異なれば `AnalyzerVersionMismatchError` を raise して非ゼロ終了。
- `compat_match.*` は major.minor 一致で OK、patch 相違は `RuntimeWarning`。

#### (b) エンティティ知識ベース

一級出力として downstream でそのまま消費することを想定。人が読む / LLM が引く / Git で diff する想定。

```
output_dir/entities/
├── manifest.json                             # 全エンティティの状態とメタ
├── ORG__スカラー商事_1df045.md
├── PERSON__田中_53b9a0.md
├── PERSON__佐藤_b9894c.md
└── PRODUCT__アルファ合成_11f89b.md
output_dir/index.md                            # manifest.json を読んだエンティティ目次
```

**ファイル名**: `{NER_LABEL}__{sanitized_name}_{sha6}.md`。`sha6` は元エンティティ名の SHA256 先頭 6 桁で、`A/B` と `A\B` のような衝突 (sanitize で `A_B` に潰れる) を防ぎます。

**`manifest.json`**:

```json
{
  "version": 1,
  "generated_at": "2026-04-21T14:14:33Z",
  "ginza_model": "ja_ginza_electra@5.2.0",
  "llm_model": "qwen3:8b@5bd05350f7c9a2c0",
  "prompt_template_version": "v1",
  "entries": {
    "ORG__スカラー商事": {
      "entity_name": "スカラー商事",
      "ner_label": "ORG",
      "source_hash": "sha256:...",
      "status": "success",               // success | failed | budget_skipped
      "last_attempt_at": "...",
      "mention_count": 5,
      "chunk_count": 3,
      "chunk_ids": ["6cac241cf928", ...],
      "cooccurring_entities": [{"name": "田中", "ner_label": "PERSON"}, ...],
      "failure_reason": ""
    }
  }
}
```

**wiki ページ本体** (`ORG__スカラー商事_1df045.md`):

YAML frontmatter (manifest の該当エントリを冗長記録) + 本文。LLM 応答中の `^---$` 行は書き出し時に fence (`\---`) されるため、frontmatter 境界は壊れません。

```
---
entity_name: スカラー商事
ner_label: ORG
status: success
last_attempt_at: '...'
chunk_ids: [...]
source_hash: ...
ginza_model: ja_ginza_electra@5.2.0
llm_model: qwen3:8b@5bd05350f7c9a2c0
prompt_template_version: v1
mention_count: 5
chunk_count: 3
ai_verification_status: unverified
cooccurring_entities: [...]
---

# スカラー商事 (ORG)

<!-- summary-begin -->
(LLM 生成の日本語サマリ)
<!-- summary-end -->

## 出現チャンク
- `6cac241cf928` (01_news.txt)
  > スカラー商事は本日、...
```

---

## インストール

### 前提環境

- **Python 3.11 または 3.12**。3.13 以降は `ja-ginza-electra` 依存の `tokenizers<0.14` に prebuilt wheel が無く、Rust ソースビルドも失敗するため `pyproject.toml` で `>=3.11,<3.13` に固定しています。macOS では `brew install python@3.11` で導入できます。
- macOS / Linux (M1/M2/Intel) で動作確認。Windows は ChunkRecord のパス正規化 (`Path(...).relative_to(...).as_posix()`) では対応していますが、e2e は未検証です。

### 依存インストール

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m spacy validate                # ja_ginza_electra の導入確認 (✔ が出れば OK)
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

---

## 使い方

### 最短経路

```bash
# 既定 (Anthropic, claude-haiku-4-5) で処理
chunking ingest samples/ out/

# Ollama ローカルモデルを明示指定
chunking ingest samples/ out/ --llm-backend ollama --llm-model qwen3:8b

# 検索 (ベースライン)
chunking query out/ "合併の背景" --top-k 5

# 健全性チェック
chunking lint out/
```

### `chunking ingest`

| フラグ | 既定 | 効果 |
|---|---|---|
| `--skip-wiki` | off | wiki / manifest / index.md を生成しない。`chunks.jsonl` + `vocab.npz` + `analyzer.json` のみ必要な用途 (開発ループで LLM 無しにイテレートしたい時) に高速。|
| `--max-llm-calls N` | 無制限 | LLM 呼び出しの上限 (entity 試行のみ、pre-flight は budget から除外)。`mention_count DESC → chunk_count DESC → entity_name ASC` の決定論順序で消化するため、予算を絞ると高価値エンティティから wiki が入ります。|
| `--force-regenerate` | off | source_hash 一致でも全 wiki を再生成。プロンプト変更時などに使用。|
| `--retry-failed` | off | 前回 `status=failed` / `budget_skipped` のみを再試行。`--force-regenerate` と同時指定時は force が優先され、retry は無視 + 警告。|
| `--llm-backend {anthropic,ollama}` | `anthropic` | LLM バックエンド。`anthropic` は `ANTHROPIC_API_KEY` 必須。|
| `--llm-model MODEL` | バックエンド既定 | 選択したバックエンドに渡すモデル名 (下表)。|
| `--format {human,json}` | `human` | `json` 指定時は `IngestResult` サマリが stdout に出る (agent 連携用)。|
| `--quiet` | off | identity banner と human success 行を抑止。|

#### `--llm-model`

| バックエンド | 未指定時の既定 | 例 |
|---|---|---|
| `anthropic` | `claude-haiku-4-5` | `--llm-model claude-sonnet-4-5` |
| `ollama`    | `qwen2.5:7b-instruct-q4_K_M` (Ollama 側で pull 済みが必要) | `--llm-model qwen3:8b` |

`--llm-model` は `--llm-backend` で選んだバックエンドに透過的に渡されます。バックエンドごとに独立したフラグを分ける代わりに、「バックエンド × モデル名」の 1 ペアで指定する設計です。

#### 代表的なユースケース

```bash
# 1. 開発ループ: LLM 無しで chunks/vocab/analyzer の挙動だけ追う
chunking ingest samples/ out/ --skip-wiki

# 2. CI: Ollama でオフライン & 予算キャップ付きで wiki も含めて確認
chunking ingest samples/ out/ --llm-backend ollama --llm-model qwen3:8b \
    --max-llm-calls 10 --format json --quiet

# 3. プロンプト改定後の一括更新
chunking ingest samples/ out/ --force-regenerate

# 4. 前回予算切れ / ネットワーク障害で失敗した分だけリカバリ
chunking ingest samples/ out/ --retry-failed

# 5. 機械可読サマリだけ取る (エージェントがパイプ受け)
chunking ingest samples/ out/ --format json --quiet | jq .exit_code
```

ingest 完了時に `output_dir/ingest_result.json` を必ず書き出します。JSON 形式指定時はこれが stdout にもそのまま出ます。

### `chunking query`

```bash
chunking query out/ "合併の背景" --top-k 5
chunking query out/ "合併の背景" --format json | jq '.[0].chunk_id'
```

| フラグ | 既定 | 効果 |
|---|---|---|
| `--top-k N` | 5 | 返すチャンク数。`N <= 0` は argparse エラー。|
| `--format {human,json}` | stdout が tty なら `human`、それ以外は自動で `json` | `human` は `[rank] chunk_id=... score=... / keywords: ... / text: ...` 形式、`json` は `QueryHit[]`。|

cosine 計算は `vocab.npz` と同じ analyzer をランタイムに再現した上で行われるため、`ingest` 時と同じ解析器設定が必要です (`analyzer.json.strict_match` で検証)。**クエリが OOV / ゼロトークンの場合は exit 4** を返し、「一致なし」と区別できます。

### `chunking lint`

```bash
chunking lint out/
chunking lint out/ --format json | jq '.summary'
```

| フラグ | 既定 | 効果 |
|---|---|---|
| `--format {human,json}` | `human` | `json` 時は stdout に JSON summary + `lint.json` サイドカーを出力。|
| `--pairwise-threshold N` | 2000 | pairwise 比較を打ち切るチャンク数。大きなコーパスでの O(N²) 爆発回避。|
| `--duplicate-cosine F` | 0.95 | 重複判定のコサインしきい値。|
| `--degenerate-l2 F` | (内部既定) | 縮退とみなす L2 下限。|
| `--degenerate-nnz N` | (内部既定) | 縮退とみなす nnz 下限。|
| `--levenshtein-ratio F` | (内部既定) | 表記近似エンティティ検出の SequenceMatcher 比率。|
| `--target-chunk-chars N` | 500 | ingest の `target_chars` と揃える。|
| `--max-chunk-chars N` | 1500 | ingest の `max_chunk_chars` と揃える。|

レポートは `lint.md` (常時) と `lint.json` (`--format json` 時のみ) に出力。下記 3 段階を使い分けます:

- **致命 (fatal)**: 契約違反。例: `chunks.jsonl` と `vocab.npz` の行数不一致。
- **警告 (warning)**: 品質を下げる可能性。例: 縮退チャンク (極端に少ない nnz / 低 L2)、重複チャンク、孤立 wiki (manifest にあるが .md が無い / 逆)。
- **情報 (info)**: 人手判断が要る示唆。例: 表記近似エンティティ候補 (「田中」「田中太郎」)。

---

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

---

## 動作保証と設計上の選択

### 破壊的全再生成

`ingest` は毎回 `output_dir` 丸ごと staging → rename で置き換えます。**差分 ingest は提供しません**。ただし `manifest.json` の `source_hash` 一致で wiki 生成のみスキップされるため、高コストの LLM 呼び出しは実質キャッシュされます。

### source_hash によるキャッシュ

wiki 再生成の要否は次の 3 要素を連結した sha256 で決定します:

1. 該当エンティティが出現するチャンク内容 (char_start/end と text)
2. `analyzer.json` のハッシュ (strict_match 変更でキャッシュ全滅)
3. `PROMPT_TEMPLATE_VERSION` (プロンプト改定でキャッシュ全滅)

一致なら前回の `.md` を staging にコピーして LLM を呼ばず、manifest の `status` を `success` のままにします。`--force-regenerate` はこのチェックを無条件に飛ばします。

### Atomic staging swap

書き込みはすべて `<output_dir>.staging/` に寄せ、最後に `os.replace(staging, output_dir)` で入れ替えます。クロスファイルシステム rename が失敗した場合は `shutil.copytree` にフォールバックしつつ、既存 `output_dir` は `.backup` に退避。`exit_code=6` (systemic abort) 時は staging の manifest を `<output_dir>.failed/entities/manifest.json` に退避してから staging を削除します。

途中終了 (Ctrl-C / 例外) しても `try/finally` で staging は必ず掃除されます。

### Deterministic ordering

- **chunk_id**: POSIX 相対パス + char offset + text の sha256 (前 12 桁) で決まり、同一入力で同じ `chunk_id`。
- **row_index**: `all_chunks` をコーパス全体で enumerate した順序。
- **LLM 呼び出し順**: `mention_count DESC → chunk_count DESC → entity_name ASC`。予算が効いた際に高価値エンティティから確実に wiki が生えます。
- **Manifest entries**: `sort_keys=True` + canonical JSON。

### 正規化の契約

`normalize_text` (`normalize.py`) は次を必ず行います:
1. NFKC
2. CRLF / CR → LF
3. 各行末の trailing whitespace を strip
4. 連続空白を collapse
5. 全文を `.strip()`

`analyzer.json.strict_match.normalization` に `{nfkc, lf_only, strip_trailing, collapse_spaces}` をすべて `true` として記録し、ingest 時と query 時で同じ契約が適用されたことを検証します。

### LLM 予算 / 信頼性

- **pre-flight**: `resources/preflight_prompt.txt` を 1 回実行して疎通確認。これは `--max-llm-calls` の予算に含めず `preflight_calls` に別計上します。
- **retry**: `LLMRetryableError` (タイムアウト / rate limit / 5xx) は最大 3 回までリトライ。`LLMPermanentError` (認証エラー / モデル不在 / 4xx) は即座に `status=failed`。
- **systemic failure abort**: 5 件以上試行した時点で failure ratio > 50% なら `LLMPermanentError` を上位に raise して `exit_code=6`。staging は `.failed/` に退避。
- **Ollama timeout**: `OllamaLLMClient` は `timeout_seconds` 既定 60 秒で、ハング時は `LLMRetryableError` に変換されます。
- **qwen3 対応**: 既定で `think=False` を渡す。無効にしたい場合は `OllamaLLMClient(think=True)` (現在 CLI からは expose していません)。

### Ginza 5.2 × spacy 3.8 互換シム

`ja-ginza-electra 5.2.0` 本体は spacy 3.5 前後を想定した古い設定で配布されています。`analyzer.py` 側で次を runtime に行うことで spacy 3.7〜3.8 + ginza 5.2 を動かします:

1. `spacy.load(..., config={"components": {"compound_splitter": {"split_mode": split_mode}}})` で `None is not <class 'str'>` の Config 検証エラーを回避。
2. `spacy.tokens.Token.set_extension("ne", getter=...)` を動的に登録。getter は `token.ent_iob_ + ginza.ENE_ONTONOTES_MAPPING[token.ent_type_]` から `B-ORG` / `I-PERSON` / `None` を派生させます。plan の BIO-walk ロジックはそのまま流用できます。

Ginza 本体が上記 2 点を自前で解決するバージョンに上がったら、shim は削除してください。

---

## 既知の限界

- **TF-IDF 単独の言い換えクエリ脆弱性**: 同義語や表現違いのクエリ (例: 「合併の背景」→「経営統合の経緯」) では関連チャンクを取りこぼすことがあります。本番 RAG 検索は dense retrieval (埋め込みモデル) に委ねる前提です。
- **固有名詞の表記ゆれ未統合**: 「スカラー」「Scalar」「スカラ商事」は NER が別エンティティとして扱い、別 wiki ページになります。`lint` で表記近似候補を列挙しますが、統合は手動判断です。
- **TF-IDF IDF 安定性**: 数十チャンク規模のコーパスでは IDF 統計が不安定になります (文献通り)。本番コーパス規模で運用してください。
- **Ollama 日本語要約品質**: モデル依存。本番品質は Anthropic を推奨。
- **単一プロセス前提**: `log.md` / manifest / staging に同時書き込みする複数 ingest 実行はサポートしません。
- **Ginza 互換シム**: 上記 shim は Ginza 5.2 世代の runtime 事情に依存。Ginza 本体更新時は要見直し。
- **embedding / BM25 / vector DB 書き込みは対象外**: `chunks.jsonl` のスキーマ安定性のみ保証します。後続パイプラインは別リポジトリで扱います。

---

## ライセンス

MIT
