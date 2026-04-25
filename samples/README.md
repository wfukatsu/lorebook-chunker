# samples/

日本語 RAG pipeline のエンドツーエンド検証と、`scripts/bench.py` の RAG 検索評価ベンチマークで使用するサンプル corpus と fixture を配置しています。

## コーパスファイル

| ファイル名 | ジャンル | サイズ |
|---|---|---|
| `01_news.txt` | ビジネスニュース (企業合併) | ~830 B |
| `02_tech.txt` | 技術ドキュメント (ベータプロジェクト) | ~610 B |
| `03_interview.txt` | インタビュー / 対談 | ~660 B |
| `04_fiction.txt` | 小説抜粋 | ~450 B |

**意図的に小規模** (合計 ~2.5 KB) に抑えています。**本番品質の検索評価ではなく、pipeline の動作確認 / regression 検出** が目的です。大規模 corpus での挙動確認は `test_optics/` (別ディレクトリ、~1.2 MB) や `tests/fixtures/large_corpus_generator.py` (RUN_LARGE=1 gated) を使用してください。

固有名詞は意図的に多めに配置しています (`田中` / `佐藤` / `スカラー商事` / `東京` / `大阪` / `プロジェクト` 等) — NER 集約 (U1 の `aggregate_entities`) / wiki 生成 / TF-IDF キーワード抽出の各経路がサンプル上でも通ることを保証します。

## fixture ファイル

### `expected.yaml` — スモークテスト fixture

`tests/test_smoke.py` が corpus 整合性 (query / wiki pages / manifest entries) を検証するために消費します。substring match ベースで、query 結果 / wiki content / manifest 構造の最小保証を表現します。

### `qrels.jsonl` — RAG 検索評価 gold set (U9)

`scripts/bench.py` が `recall@k / MRR@10 / nDCG@10` を計算するための正解集合。JSONL 形式、1 行 1 クエリ:

```jsonl
{"qid": "q1", "query": "合併の背景", "relevant": [{"chunk_id": "08ed5bc2ff40", "grade": 2}]}
```

- `grade` は TREC 慣習 (0 = 非該当、1 = 関連、2 = 強く関連)
- 現在 8 クエリ収録 (業界合併 / 新社名 / ベータプロジェクト技術スタック / 対談経営統合 / 東京大阪統合 / 小説情景 / QA 担当 / 株価合併発表)
- chunk_id は **ingest で content-dependent に生成** されるため、`chunk_size` / `overlap` を変えると chunk_id が変わります。`qrels.jsonl` は c256 / c512 ベースでハンドオーサリングされており、他の config では一部 chunk_id が不在になる可能性がありますが、その場合は該当 query の recall が 0 に degrade するのみで crash はしません

詳細は README の「RAG 検索評価ベンチマーク」セクションを参照。

## 歴史的メモ

本ディレクトリは元々 [docs/plans/2026-04-21-001-feat-japanese-rag-chunking-plan.md](../docs/plans/2026-04-21-001-feat-japanese-rag-chunking-plan.md) Unit 11 に基づき「6 本 ~30,000 字を執筆する」方針でしたが、実装過程で「まず pipeline 正常性を保証する最小 corpus」に絞り込まれました。大規模 corpus での評価は `test_optics/` で実施する運用です。
