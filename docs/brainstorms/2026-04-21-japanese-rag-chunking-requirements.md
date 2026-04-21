---
date: 2026-04-21
topic: japanese-rag-chunking
---

# 日本語 RAG 向けチャンク化 + TF-IDF + 固有名詞 Wiki スクリプト

## Problem Frame

日本語文書を RAG 用コーパスに変換するパイプラインを Python で構築したい。Ginza による文境界を尊重した可変長チャンク化、TF-IDF による疎ベクトル化、加えて Karpathy の "LLM Wiki" 思想から着想を得た固有名詞中心の要約ページ生成を一体で扱う。

従来のチャンク化スクリプトは「chunks.jsonl を吐いて終わり」になりがちで、(a) コーパス健全性の確認が後回し、(b) 固有名詞レベルでの「何が書かれているか」が失われる、という2点が課題。本スクリプトはこれを `ingest` / `query` / `lint` の3操作と、エンティティ wiki の自動生成で解消する。

**ツールのアイデンティティ**: 本スクリプトは **RAG 用コーパスの前処理 + エンティティ知識ベース生成ツール**。出力は以下の2系統で、それぞれ異なる消費者を想定する:

1. **chunks.jsonl + TF-IDF 疎ベクトル + analyzer.json**: RAG 検索器の入力（TF-IDF 検索は v1 のベースライン、将来 dense vector 化または下流 vector store へ供給する前提）。本ツールの `query` サブコマンドはコーパス整形時の確認用であって、本番検索用ではない。
2. **エンティティ wiki (entities/*.md + index.md)**: **一級出力**として扱う。想定消費者は (a) コーパス内容を俯瞰したい開発者、(b) 将来 LLM 応答生成時にエンティティ文脈を事前注入したい下流 RAG システム、(c) エンティティ知識のドキュメントとして人間が読む用途。v1 では自動注入機構は実装しないが、出力構造（frontmatter スキーマ、ファイル名、index.md フォーマット）は下流消費者の契約として安定的に維持する。

## Scope Overview

```
raw .txt files  ──▶  [ingest]  ──▶  chunks.jsonl  (chunk_id, text, tfidf sparse, top_keywords[], entities[])
                        │             analyzer.json  (Ginza model + POS filter config)
                        │             vocab.npz     (TF-IDF vocabulary + IDF + transformer)
                        │
                        ├──▶  entities/{entity}.md  (LLM要約 + 該当 chunk_id 一覧、frontmatter に共起語)
                        │
                        └──▶  index.md, log.md

query "..."     ──▶  [query]   ──▶  top-k chunks (TF-IDF cosine; analyzer.json を復元して同一前処理)

                     [lint]    ──▶  健全性レポート (空チャンク/重複/縮退/孤立/表記近似エンティティ)
```

## Requirements

**チャンク化 (Chunking)**
- R1. `ja_ginza_electra` モデルで文境界分割と POS/NER を行う。文をまたいで機械的に切らない。**NER ラベルは OntoNotes5 を用いる**が、`ja_ginza_electra` は `token.ent_type_` / `ent.label_` では Sekine 拡張固有表現 (`Person`, `Company`, `City`, …) を返すため、OntoNotes5 ラベルは `token._.ne` セカンダリ属性で取得すること（本要件内で「NER ラベル」と書いた場合は常に OntoNotes5 側を指す）。利用した Ginza パッケージ版数・spaCy 版数・モデル名・モデルチェックサム・`sudachipy` 版数・`sudachidict_*` パッケージ名と版数を ingest の成果物 (`analyzer.json`) に記録する。
- R2. チャンクサイズは**文字数目標 + 文境界尊重**で決める。デフォルトは **目標 500 文字 / オーバーラップ 100 文字**。両値は CLI 引数と設定ファイルの両方で上書き可能。
- R3. 1文が目標チャンクサイズを超える場合:
  - 長さが `max_chunk_chars`（デフォルト **1500 字**、設定可能）以下であればその文を単独 1 チャンクとし警告ログを出す（破壊的に切らない）。
  - `max_chunk_chars` を超える場合は読点 (`、`) / 改行 / コードブロック境界など副次的な境界でソフト分割する。ソフト分割も不可能なら文単位で出力し `lint` が致命度「警告」で検出する。
- R4. オーバーラップは「直前チャンク末尾から N 文字ぶんを次チャンク先頭に重複付与」で実装し、重複開始位置は文境界にスナップする。境界解決ルール:
  1. `(current_end - N)` 位置から過去方向に最も近い文境界を採用する。
  2. 窓幅 `[N/2, 2N]` の範囲に境界が無ければ、オーバーラップを 0 にし `log.md` に記録する。
  3. 文境界が窓より遠い側にあっても、2N を超えてまで拡張しない。

**TF-IDF / キーワード抽出 / 前処理アナライザ**
- R5. コーパス全体で TF-IDF を学習し、各チャンクに対応する**疎ベクトル**を保存する（検索用）。
- R6. 各チャンクに対し、TF-IDF スコア上位 **N 語** をキーワードとしてメタデータに付与する（デフォルト N=10、設定可能）。
- R7. TF-IDF の前処理として、Ginza の品詞情報を使って助詞・助動詞・記号を除外し、名詞・動詞・形容詞の基本形を語彙単位とする。Sudachi の分割モード、`sudachidict_*` パッケージ名・版数、ストップワードリスト、lemma 化規則、**テキスト正規化設定（NFKC + 改行 LF 統一 + 末尾空白除去 + 連続空白折りたたみ — R18 の `source_hash` 計算で用いるものと同一定義）**はすべて `analyzer.json` に書き出し、`query` は同じ設定でクエリ文をトークナイズする。`query` 時は `analyzer.json` のバージョンが `ingest` 時と一致しなければ非ゼロ終了する。
- R8. TF-IDF 語彙・IDF ベクトル・変換器・`analyzer.json` は `ingest` 完了時に成果物として保存し、`query` サブコマンドで再利用できること。
- R8b. `ingest` は **output_dir を全再生成する破壊的操作**とする。既存の `chunks.jsonl` / `vocab.npz` / `analyzer.json` は上書きする（増分 ingest は第一版では非対象、Scope Boundaries 参照）。ただし `entities/*.md` に対する LLM 呼び出しのみ R18 の増分判定でスキップ可。

**CLI 操作 (ingest / query / lint)**
- R9. `ingest <input_dir> <output_dir> [--skip-wiki] [--max-llm-calls N] [--force-regenerate] [--retry-failed]`: 入力ディレクトリ内の `.txt` を UTF-8 で読み取り、チャンク化・TF-IDF・エンティティ wiki までを一括生成する。エッジケース:
  - 入力ディレクトリに `.txt` が 0 件 → 非ゼロ終了コードで終了。
  - 空ファイル / 空白のみのファイル → 警告を出してスキップ、`log.md` に記録。
  - UTF-8 復号失敗 → 警告を出してスキップ、`log.md` に記録。
  - `--skip-wiki`: エンティティ wiki の生成 (R13-R18) をスキップし、チャンク化・TF-IDF・lint 材料のみを生成する。LLM 未接続の CI 環境向け。
  - `--help` および `query --help` の出力に、本ツールが「RAG 前処理・コーパス健全性・観察用ツールであり本番検索用ではない」旨の短いアイデンティティバナーを表示する（利用者の誤用防止）。
- R10. `query <output_dir> "<検索文>" [--top-k K]`: `analyzer.json` と保存済みの TF-IDF 語彙を使って検索文をベクトル化し、コサイン類似度で上位 K チャンクを返す（デフォルト K=5）。必要な ingest 成果物（TF-IDF 語彙 / IDF / `chunks.jsonl` / `analyzer.json`）が `output_dir` に存在しない・壊れている・バージョン不整合の場合は非ゼロ終了コードで「先に `ingest` を実行してください」を出力する。
- R11. `lint <output_dir>`: コーパスの健全性を検査し、レポートを標準出力と `output_dir/lint.md` に書き出す。致命度は **致命** / **警告** / **情報** の3段階。検査項目は最低でも以下:
  - **致命**: `analyzer.json` / `vocab.npz` / `chunks.jsonl` のスキーマ不整合・バージョン不整合 / 空コーパス（チャンク 0 件）
  - **警告**: 空チャンク / 1文のみのチャンク / 目標の 2 倍以上の長さのチャンク / `max_chunk_chars` 超過チャンク / 重複・ほぼ重複（テキスト一致 or コサイン > 0.95）チャンク / TF-IDF ベクトルが L2 ノルム < 1e-6 もしくは非ゼロ要素数 < 3 のチャンク（定義は planning で微調整可） / 実チャンクから参照が消えた孤立エンティティ wiki ページ / **wiki 生成数が絶対値で < 3** かつ **総チャンク数 < 20** の場合（小コーパスでの wiki 機能失効の早期検知。Zipf 分布で当然起こる「多数の1回出現エンティティが閾値未満」だけでは警告しない）
  - **情報**: **表記近似エンティティ対**（v1 は Levenshtein 比 > 0.85 のみ、読みがな一致は非対象）を列挙
  - `lint` 終了コードは、致命 0 件なら 0（警告/情報のみでも 0）、致命 1 件以上で非 0。
- R12. 3サブコマンドはすべて終了コードで成否を区別する（lint は警告のみでも 0、致命エラーで非 0）。

**エンティティ Wiki (固有名詞 LLM 要約)**
- R13. ingest 時に Ginza の OntoNotes5 NER ラベル（R1 のとおり `token._.ne` 経由で取得）で固有名詞を抽出する。デフォルト対象ラベルは `PERSON` / `ORG` / `LOC` / `PRODUCT`。対象ラベルは設定で追加・削除可能（例: `EVENT`, `WORK_OF_ART`, `FAC` を追加）。抽出結果は各チャンクメタデータの `entities[]` に `{name, ner_label, char_start, char_end}` 構造で格納する。
- R14. しきい値を満たすエンティティに対してのみ LLM 要約ページを生成する。デフォルト: `min_mentions=3`（コーパス全体での最低出現回数）かつ `min_chunks=2`（出現するチャンク数）。両値は設定可能。コーパスが極端に小さい場合（総チャンク数 < 10 等）は R11 の lint で「wiki 未生成率」警告を通じて利用者に設定変更を促す。
- R15. 各エンティティ wiki ページは Markdown で、frontmatter + 本文構造:
  - frontmatter: `entity_name`, `ner_label`, `mention_count`, `chunk_count`, `source_hash`（後述）, `ginza_model`, `llm_model`, `cooccurring_entities: [{name, ner_label}, ...]`（共起エンティティを `{name, ner_label}` オブジェクト配列で格納。Markdown 相互リンクは作らない — 表記統合が未実装のためリンク先正当性を保証できないため。ラベル付きにすることで同名別ラベル（`PERSON`「富士」vs `LOC`「富士」等）の曖昧性を frontmatter レベルで解消）
  - ファイル名は `entities/{ner_label}__{sanitized_entity_name}.md` とし、同名別ラベルの衝突を防ぐ（sanitize は OS セーフな文字集合への置換）
  - 本文: エンティティ名・NER ラベル、LLM による 2〜4 文の要約（該当チャンクのテキストを LLM に渡して生成）、出現チャンク ID のリストと、各チャンクでの前後数十文字のコンテキストスニペット
- R16. `index.md` を生成する。NER ラベル別セクションで、エンティティ名の五十音/アルファベット順一覧 + 各ページへの相対リンク。
- R17. `log.md` を生成する。ingest 実行ごとに 1 エントリ追記（日時、入力ファイル数、生成チャンク数、生成 wiki 数、スキップされたエンティティ数、LLM 呼び出し回数、**LLM 合計トークン数（prompt + completion 合算、バックエンド別に記録）**、Ginza モデル版数、LLM モデル版数）。金額換算は計測責務から除外し、トークン数を記録して利用者側で換算する。
- R18. LLM 呼び出しは**増分更新**を優先する。前回 ingest からエンティティの実態が変化していなければ再生成しない。判定キー `source_hash` は以下の合成ハッシュ:
  - `entity_name` + `ner_label`（同一チャンク集合を共有する別エンティティが衝突しないため）
  - 出現チャンクの**正規化テキスト**を sorted で連結したハッシュ（正規化は NFKC + 改行 LF 統一 + 末尾空白除去 + 連続空白折りたたみ。この正規化定義は `analyzer.json` にも記録する）
  - `analyzer.json` 内容のハッシュ（Sudachi モード/辞書・POS フィルタ・stopwords 変更を検知）
  - Ginza モデル版数・LLM モデル版数・プロンプトテンプレート版数
  
  Ollama の LLM モデル版数は `{model_tag}@{sha256_of_manifest}` 形式で記録し、タグ維持のまま再 pull された場合も検知する。ingest は実行中に `output_dir` を破壊的上書きする前に、既存の `entities/manifest.json`（または frontmatter）から前回 source_hash と成否状態をメモリに読み込む。`--force-regenerate` フラグで全再生成。
- R18b. **失敗エンティティの再試行**: 前回 ingest で LLM 呼び出しに失敗したエンティティは、`source_hash` が同じでも次回 ingest で再試行対象とする。`--retry-failed` フラグで「source_hash 未変更 + 前回成功」のものも含めて再試行可能。

**LLM バックエンド**
- R19. 共通インターフェース `LLMClient` を定義する。必須メソッド: `generate(prompt: str, max_tokens: int) -> GenerateResult`。`GenerateResult` はテキスト、入出力トークン数、モデル識別子、finish_reason を含む。バックエンド差（JSON モード、ストリーミング、レートリミット分類）はアダプタ側で吸収し、本体は `retryable_error` / `permanent_error` の分類のみ受け取る。
- R20. 第一版同梱実装:
  - **Anthropic** (`claude-haiku-4-5` デフォルト) — 本番利用を想定、既定バックエンド。
  - **Ollama** (ローカル、モデル名は設定で指定) — 外部 API 課金ゼロの開発・検証用。
  - **OpenAI** は `LLMClient` インターフェースを介した**プレースホルダ実装のみ**（設定選択時に「未実装」エラーで即終了）。実装は実消費者が現れた時点で追加する。
  - API キーは環境変数 (`ANTHROPIC_API_KEY` 等) で受け取り、コード・設定ファイルにハードコードしない。デフォルトモデル名は実装着手時の SDK で疎通確認し、利用不可なら最新公開スナップショットに差し替え。
- R21. バックエンドは設定ファイルまたは CLI フラグ `--llm-backend {anthropic,ollama}` で切替可能。未指定時のデフォルトは Anthropic。OpenAI は CLI enum から除外し、実装が揃い実消費者が現れた時点でフラグ値を追加する（現状は R19 の `LLMClient` インターフェース準拠確認用プレースホルダのみ）。
- R22. LLM 呼び出しの失敗ハンドリング:
  - **pre-flight**: ingest 開始時に 1 回だけ疎通呼び出しを行い、認証・ネットワーク不通は即時に非ゼロ終了でユーザに通知する。プロンプト形状は実際の要約プロンプトに近い（同一 system prompt・同程度の max_tokens）とし、短すぎる「OK 返答のみ」プロンプトで通るが本番プロンプトで失敗するケース（org レベルの max_tokens 制限・content policy 等）を検知可能にする。`--max-llm-calls 0` 指定時・`--skip-wiki` モード（R9 参照）時は pre-flight をスキップする。プロンプト 1 回分は `--max-llm-calls` に含める。
  - **per-call retry**: 失敗時は指数バックオフで最大 3 回リトライ。`Retry-After` ヘッダがあれば尊重。
  - **per-entity fallback**: 3 回失敗したエンティティは wiki 生成をスキップし `log.md` に「failed」状態で記録（R18b で次回再試行対象）。ingest 全体は止めない。
  - **systemic failure detection**: 試行を終えた（成功 or 最終失敗）エンティティ数が **最低 5 件**に達した後、`failed / (succeeded + failed)` が **>50%** となった時点で ingest を中止し非ゼロ終了する（budget でスキップされたものは分母に含めない。pre-flight 失敗は即時終了に直結するため分母外）。API キー無効・地域制限等のシステム故障の早期検知。
  - **budget guard**: `--max-llm-calls N` オプションで ingest 全体の LLM 呼び出し回数を上限制限。処理順は `(mention_count DESC, chunk_count DESC, entity_name ASC)` に固定し、高頻度エンティティから優先的に消化する（残り予算で最も価値の高いエンティティを生成する意図）。超過時は未呼び出しエンティティを「budget_skipped」状態で `log.md` に記録して正常終了（R18b 自動再試行の対象にはしない。`--retry-failed` で明示再試行）。
  - **R18b との優先順序**: 同一 ingest 内で budget が逼迫した場合、「前回 failed の再試行」と「新規 eligible エンティティ」は同じキューに合流させ、上記の mention_count 順で処理する。

**サンプル文章 (テストデータ)**
- R23. 以下 6 ジャンルの**架空・自作**日本語文章を同梱する。既存の実在する会社名・人名・地名は使用しない。合計約 **30,000 字**（TF-IDF IDF 統計の最低限の安定性のため、当初の 18,000 字から拡張）。
  - (1) ビジネスニュース記事（架空会社の合併/決算、約 5,000 字）
  - (2) 技術ドキュメント（架空の分散 DB 製品チュートリアル、箇条書き・コードブロック含む、約 6,000 字）
  - (3) インタビュー/対談（複数話者、敬称付き、約 5,000 字）
  - (4) 旅行エッセイ（架空地名、描写的、約 5,000 字）
  - (5) 百科事典風記事（架空の歴史上人物、エンティティ相互参照が密、約 4,500 字）
  - (6) 小説抜粋（独白・会話、エンティティ薄、約 4,500 字）
- R24. サンプルは `samples/` 配下に `01_news.txt` … `06_fiction.txt` として置き、ingest の既定入力対象として選択可能にする。
- R25. サンプル受入基準: 執筆後、日本語ネイティブのレビュアー（当面は作成者自身）が以下をチェックする:
  - 実在する会社名・人名・地名との偶発的衝突が無い
  - 文章が自然に読める（機械的な繰り返しや不自然な言い回しが無い）
  - 各ジャンル想定の固有名詞密度（ジャンルごとに目視）
- R26. 自動テストに最低限の動作確認（スモークテスト）を含める: サンプルを ingest → 既知クエリで query → 期待チャンクが top-k に含まれる、を pytest で検証。期待結果 fixture はサンプルと対になる `samples/expected.yaml` に置き、サンプル改訂時に併せて更新する運用とする。

## Success Criteria

**パイプライン形状（出力ができるか）**
- サンプル 6 本を `ingest samples/ out/` 1 コマンドで処理完了し、`chunks.jsonl` / `vocab.npz` / `analyzer.json` / `entities/*.md` / `index.md` / `log.md` が揃って生成される。
- デフォルトパラメータ（500 字目標）で、チャンク長の中央値が **400〜600 字**、最大長が **800 字未満**（R3 の単独長文チャンクを除く）に収まる。
- `lint out/` が R11 の全検査項目を実行でき、サンプルデータに対して致命エラー 0 で終了する。

**検索品質（ユーザーが困らないか）**
- **語彙一致クエリ**: `query out/ "合併の背景"` で、ニュース記事 (#1) 由来の関連チャンクが top-5 に含まれる。
- **言い換えクエリ**: `query out/ "経営統合の経緯"` でも、ニュース記事 (#1) の関連チャンクが top-10 に含まれる。**入らない場合は合格条件の代替として**、`README.md`（または `--help` バナー）に「言い換えクエリでは関連チャンクを取りこぼすことがある。本番 RAG 検索は下流の dense retrieval に委ねること」の注意書きを追記することを必須とする（利用者に限界を明示する）。

**エンティティ Wiki の価値検証（書いた意味があるか）**
- 生成された wiki ページは全て frontmatter の `mention_count >= min_mentions` かつ `chunk_count >= min_chunks` を満たす（構造的検査、数値目標ではない）。
- 生成された wiki のうち任意 5 件を作成者が読み、LLM 要約の事実誤り・ハルシネーションが各ページ **0〜1 箇所** に収まる（事実は該当チャンクのテキスト内で検証可能）。
- **Remediation path（基準超過時の対応）**:
  - サンプルでページ平均 2 箇所以上のハルシネーションが観測された場合、プロンプトテンプレートの改善で基準達成を試みる（最大 3 回の prompt 反復）。
  - 改善後も基準未達なら、wiki 機能は v1 でも一級出力として生成を継続する（`--skip-wiki` でユーザがオプトアウト可能）が、各 wiki ページの frontmatter と本文冒頭に **「AI 生成・未検証」注意書き**を自動付与し、`README.md` および `log.md` にハルシネーション基準未達状態を明記する（下流消費者が品質状態を機械的に判断できるよう、frontmatter にも `ai_verification_status: unverified` フィールドを追加）。
  - 一級出力契約（安定した frontmatter スキーマ・ファイル名・index.md）は品質状態に関わらず維持する。

**バックエンド互換性**
- `--llm-backend ollama` で ingest が完走する（外部 API 課金なしで動作確認できる）。ローカル Ollama の日本語要約品質は Deferred 扱いだが、呼び出し自体は通ること。

## Scope Boundaries

- **非対象: 埋め込みモデル（dense vector）の生成。** TF-IDF のみ。言い換えクエリへの弱さは本ツールの識別境界として受け入れ、本番 RAG 検索は下流の vector store 側で行う前提。
- **非対象: BM25 スコアリング。** v1 はコサイン類似度のみ。
- **非対象: 増分 ingest。** `ingest` は常に `output_dir` を全再生成する破壊的操作（R8b）。エンティティ wiki の LLM 呼び出しのみ内部的に増分（R18）。入力コーパスが変わった場合はディレクトリを再生成する運用を前提とする。
- **非対象: ベクトル DB / ScalarDB への書き込み。** 出力は `jsonl` とローカルファイルのみ。連携は後続タスク。ただし `chunks.jsonl` のスキーマ（`chunk_id`, `text`, `source`, `char_offsets`, `sparse_vec`, `top_keywords`, `entities`）は将来取り込みを容易にするため安定契約として維持する。
- **非対象: HTML / PDF / Office の直接読み込み。** 入力は UTF-8 の `.txt` 前提。
- **非対象: wiki ページ間の自動的な意味統合（"同一人物の別表記を統合" のような表記ゆれ解決）。** NER が別エンティティとして返したものは別ページとする。ただし R11 の lint で類似表記候補を情報レベルで列挙する（利用者が手動で判断できるよう材料を出す）。
- **非対象: Web UI / サーバ。** CLI のみ。
- **非対象: マルチ言語対応。** 日本語専用。
- **非対象: wiki の RAG パイプラインからの自動消費機構（プロンプト注入・API 提供）。** 出力物としての wiki は一級出力だが、それを消費する機構（RAG 応答時にエンティティページを引く API、プロンプトテンプレート差し込み機構等）は v2 以降。v1 は「下流消費者が読みやすい安定フォーマットで書き出す」ところまでを扱う。
- **非対象: 読みがな (phonetic) ベースのエンティティ表記近似検出。** v1 は文字列 Levenshtein 比のみ。pykakasi 等の読み推定は依存を増やすため追加しない。
- **非対象: エンティティ wiki 生成を伴わない独立サブコマンド (`generate-wiki` 等)。** wiki は `ingest` 内でのみ生成する（`--skip-wiki` でスキップのみ選択可）。

## Key Decisions

- **チャンクサイズは文字数目標方式**: トークン目標より実装・テストが単純で、RAG 用途のデファクト。将来埋め込みモデルが固定されたらトークン方式へ差し替え可能な設計にする。
- **TF-IDF で検索用疎ベクトル + チャンクごとのキーワード両方**: 検索器とメタデータ表示の両用途を単一パスで満たし、追加コスト小。言い換えクエリに対する検索品質は本ツールのスコープ外 — 本番は下流の vector store で補う前提。
- **Karpathy の "LLM Wiki" からは操作モデル (ingest/query/lint) を採用し、LLM 要約は固有名詞にスコープ限定**: 全チャンク要約は LLM コストが線形増で膨らむため避けた。`--max-llm-calls` で上限ガードも導入済み。
- **wiki を一級出力として扱う**: 当初「観察用」と位置付けていたが、LLM コスト制御・増分ハッシュ・ハルシネーション検証など一級出力レベルの作り込みが既に揃っていること、および将来的に RAG 応答時にエンティティ情報を注入する consumer が現実的にあることから、v1 から一級出力契約（安定した frontmatter スキーマ・ファイル名規則・index.md 構造）として扱う。自動注入機構の実装は v2。
- **LLM バックエンドは Anthropic を本命、Ollama を開発ゼロコスト検証用、OpenAI はインターフェースのみ**: 3バックエンド同時実装は speculative abstraction のため v1 では避け、実消費者が出た時点で OpenAI を実装する。
- **ingest は破壊的全再生成**: 語彙・疎ベクトル・アナライザの整合性を維持する最も単純な方法。増分は将来課題。
- **R18 の増分判定は出現チャンク本文ハッシュ + モデル版数**: チャンクID の安定性に依存せず、モデル更新時は強制再生成される。
- **サンプルは全文自作 + 30,000 字**: 著作権と実在エンティティ衝突リスクを避け、TF-IDF IDF 統計の最低限の安定性（概ね 60 チャンク弱）も確保。
- **コスト追跡はトークン数まで、金額換算はしない**: プロバイダ別単価テーブルの維持コストを避ける。利用者が必要なら外部で換算。

## Dependencies / Assumptions

- Python 3.11+ と、`spacy`, `ginza`, `ja_ginza_electra`, `sudachipy`, `scikit-learn`, `scipy`, `anthropic`, `ollama`, `pytest` への依存を想定（最終ライブラリ選定は planning で確定）。OpenAI SDK は v1 ではオプショナル（未実装プレースホルダのためインポートしない）。
- Ollama は開発者のローカルに別途インストール済みである前提（スクリプトが自動導入しない）。
- ユーザは `ANTHROPIC_API_KEY` を環境変数で設定済み、もしくは `--llm-backend ollama` を使う前提。
- Ginza の NER 精度は完璧ではない（未知語・口語・テクニカル用語で誤検出・不検出あり）ことを前提とし、`lint` の孤立/表記近似検査と「wiki 未生成率」警告で事後的に検知する運用。

## Outstanding Questions

### Resolve Before Planning

（なし。ブレインストーミング内およびレビュー対応で全ての製品判断が解決済み。以下はいずれも planning 時の調査/決定で十分。）

### Deferred to Planning

- [Affects R5,R8][Technical] TF-IDF 疎ベクトルの永続化形式 — `scipy.sparse.save_npz` か、JSONL 内にインラインで `{indices, values}` を格納するか。検索性能とデバッグしやすさのトレードオフ。
- [Affects R18][Technical] `source_hash` と「前回失敗状態」の物理的な保存先 — wiki ページの frontmatter のみで完結するか、別 `entities/manifest.json` を併用するか。`--retry-failed` 実装時の I/O 効率（全 md スキャン vs 単一 JSON 読み込み）とも両立する方を選ぶ。
- [Affects R20][Needs research] Ollama バックエンドでの日本語要約品質が wiki 用途に耐えるか、モデル選定（`qwen2.5`, `gemma2` 等）。サンプル #5 で実測する。
- [Affects R13][Needs research] Ginza の OntoNotes5 ラベル（`token._.ne` 経由）が技術ドキュメント (#2) のテクニカル固有名詞（API 名・製品名・OSS 名）に対してどの程度 `PRODUCT` / `ORG` で拾うか、サンプル #2 でベースライン計測する。不足が大きい場合は planning で「ユーザー辞書注入」や「TF-IDF 上位語を補助的に固有名詞候補として扱う」などの補完策を検討。
- [Affects R23,R26][Technical] サンプル #5（架空歴史人物の百科事典風記事）は LLM 下書き → 人間校正のフローで作る想定。執筆の実作業量と品質担保方法を planning で確定。サンプル改訂時の `samples/expected.yaml` 更新方法（厳密 chunk_id 一致 vs substring 一致）も planning で決定。
- [Affects R11][Technical] TF-IDF 縮退判定の具体値（`L2 < 1e-6 もしくは nnz < 3` の妥当性）と `min_df` の値を、サンプル corpus (~60 チャンク) で実測してから微調整。
- [Affects R10][Technical] query に entity フィルタ（「X が言及されたチャンクから top-K」）を後付けするか。wiki の frontmatter に chunk_ids があるので低コストで実装可能だが、これは Scope Boundaries「wiki の RAG パイプラインからの自動消費機構」に該当する v2 機能。v1 スコープには入れない方針で planning は進め、実施時期は v2 で再検討する。
- [Affects R17][Technical] `log.md` への書き込みがマルチ ingest 実行の競合に耐える必要があるか（単一開発者ツール想定なら不要）。
- [Affects R22][Technical] pre-flight プロンプトを「本物の要約プロンプトと同形状」にするための具体例（1エンティティぶんのダミー入力を作るか、固定 fixture にするか）。

## Next Steps

- `Resolve Before Planning` は空。`/ce-plan` で実装計画へ進めます。
