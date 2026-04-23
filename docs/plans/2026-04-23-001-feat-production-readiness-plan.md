---
title: Production readiness — input handling, error taxonomy, run contract, atomic swap, ergonomics, eval
type: feat
status: active
date: 2026-04-23
deepened: 2026-04-23
origin: https://github.com/wfukatsu/lorebook-chunker/issues/1
---

# Production readiness — input handling, error taxonomy, run contract, atomic swap, ergonomics, eval

## Overview

GitHub Issue #1 (`wfukatsu/lorebook-chunker`) が提起する7項目の production-readiness ギャップ全てを1つのリリースで解消する。中身はアルゴリズム改修ではなく **オペレーショナル契約**: 入力の堅牢性、エラーの分類・exit code 契約、atomic swap の明示化、機械可読な実行レポート、セットアップの見通し、統合テストカバレッジ、RAG 検索品質のベンチマーク。既存の chunks.jsonl / entities/*.md / analyzer.json のスキーマは変更しない (下流 RAG 消費者の契約として安定維持)。

`ingest_result.json` は `run_report.json` に置き換え (破壊的変更、現在 v0.1.0 で外部消費者は実質不在の前提)。`--no-atomic` は提供せず、代わりに `--verify-swap` による SHA-256 照合のみを opt-in で追加。RAG 評価は自前ゴールドセット + ranx (optional extra) を採用し、外部データセット統合は見送り。

---

## Problem Frame

本ツールは設計上、決定論・再現性・LLM 予算制御に強い。一方で、Issue #1 が指摘するように、実運用パイプラインに組み込む際に以下の operational contract が未整備:

- **入力処理の硬直性**: `glob("*.txt")` トップレベル限定、UTF-8 のみ (`src/lorebook_chunker/ingest.py:564-567, 353-368`)。スキップされたファイルは文字列警告のみ (`ingest.py:358-360`)。
- **エラー分類の粗さ**: トップレベル `except Exception → exit=10` (`ingest.py:549-553`)。LLM エラー以外は taxonomy 未整備。
- **atomic swap の契約未定義**: `rename` → EXDEV fallback `copytree` (`ingest.py:617-622`) のみで、cross-device / 部分失敗 / クラッシュ時の挙動を文書化していない。parent-dir fsync なし。
- **実行結果契約の不足**: `ingest_result.json` は F-036 マーカーの ad-hoc 追加 (`ingest.py:794-823`)。per-phase 時間、entities_generated、skip 理由、analyzer モデル情報を欠く。
- **依存の重さ**: spaCy/GiNZA/transformers + ELECTRA 約400MB。プロファイル (minimal/fast/full) が未文書化、環境検証コマンドなし。
- **テストギャップ**: 大容量、混在エンコーディング、cross-device rename、resume-after-crash の統合テストが無い。
- **RAG 評価の欠如**: substring-match の単一スモークテスト (`tests/test_smoke.py:112-160`) のみ。recall@k / MRR / nDCG の計測機構なし。

本計画はこれらを、既存の「決定論・再現性」を崩さずに全て埋める。

---

## Requirements Trace

Issue #1 の7節を R1〜R13 として分解。U-ID は Phase 3 で採番した本計画固有。

- **R1.** `--recursive` による入力ディレクトリ再帰探索 → U2
- **R2.** `--glob "pat1,pat2"` による拡張子/パターン指定 → U2
- **R3.** `--encoding auto|utf-8|<name>` によるエンコーディング処理 → U3
- **R4.** `skipped_files.jsonl` 機械可読スキップレポート → U4
- **R5.** 構造化エラー分類 (`ConfigError`/`InputError`/`EncodingError`/`AnalyzerInitError`/`ChunkingError`/`TfidfError`/`EntityAggregationError`/`WikiGenerationError`/`AtomicSwapError`/`RunReportError`/`LLMBackendUnavailableError`) → U1
- **R6.** 安定 exit code マッピング (2/3/4/5/6/10 を後方互換維持、11-17 を新規) → U1
- **R7.** atomic swap の契約文書化 (same-fs vs cross-device、失敗状態、durability スコープ) → U6
- **R8.** `--verify-swap` による SHA-256 照合 (Issue の `--no-atomic` 提案は採用せず、代替として verification 側のみ提供) → U6
- **R9.** `run_report.json` 安定スキーマ (files_processed/files_skipped-with-reasons/chunks_generated/entities_generated/phase_durations/analyzer+model メタデータ) → U5。既存 `ingest_result.json` は削除
- **R10.** 依存プロファイル文書化 (minimal/fast/full/bench/dev) → U7
- **R11.** `--dry-run` および `doctor` サブコマンドによる環境検証 → U7
- **R12.** 統合テスト (破損入力、混在エンコーディング、resume/crash、cross-device rename、大容量) → U8
- **R13.** RAG 検索評価ベンチマーク (recall@k / MRR@10 / nDCG@10、chunking params + wiki on/off 比較) → U9

**Origin flows and acceptance examples:** Issue 本文は F-ID / AE-ID を付与していないため、本計画ではフローは `IngestRunner.run` の既存フェーズ順 (analyzer-init → chunking → tfidf → ner → wiki → swap) を基準とし、受入例は各 U-ID の Test scenarios に展開する。

---

## Scope Boundaries

- 埋め込みモデル (dense vector) 生成、BM25 スコアリングは対象外 (prior plan から継承)
- 増分 ingest (常に破壊的全再生成) は維持。resume はテスト観点のみ
- `chunks.jsonl` / `analyzer.json` / `entities/*.md frontmatter` のスキーマ変更は対象外 (下流 RAG 消費者契約)
- マルチライター並行 ingest は非対象 (現行の single-writer 仮定を維持)
- HTML / PDF / Office 直接読み込みは非対象 (テキスト入力のみ)
- ベクトル DB / ScalarDB 書き込みは非対象
- 表記ゆれの意味統合は非対象
- 外部 RAG データセット (MIRACL-ja / JQaRA / JMTEB) の統合は非対象 (benchmark は自前 gold set + 汎用 TREC qrels フォーマット)
- Windows サポートは best-effort。POSIX (Linux/macOS) がプライマリターゲット

### Deferred to Follow-Up Work

- `ingest_result.json` 互換 alias レイヤー: 削除する。将来的に外部消費者が現れた場合の再検討は別 Issue
- `renameat2(RENAME_EXCHANGE)` による Linux-only 三方向スワップ最適化: portable fallback がある限り優先度低
- 外部 Japanese RAG ベンチマーク (JQaRA, MIRACL-ja) の統合: U9 完了後に需要を見て別 Issue

---

## Context & Research

### Relevant Code and Patterns

既存実装のアンカー (repo-relative, 行番号付き):

**エラー契約**
- `src/lorebook_chunker/schema.py:119-128` — `AnalyzerVersionMismatchError`, `AnalyzerNEUnavailableError`, `ChunkFileCorruptError` (RuntimeError 派生)
- `src/lorebook_chunker/llm/__init__.py:28-41` — `LLMError` → `LLMRetryableError` / `LLMPermanentError` → `LLMPreflightError` 階層
- `src/lorebook_chunker/cli.py:13-37` — 各サブコマンドの exit code epilog (現在 INGEST/LINT/QUERY で重複記述)
- `src/lorebook_chunker/ingest.py:720-729` — `INGEST_EXIT_CODES` dict (CLI epilog と重複)
- `src/lorebook_chunker/ingest.py:304-553` — top-level try/except チェーン (exit 3/4/5/6/10)

**Atomic 書き込みの reference パターン**
- `src/lorebook_chunker/wiki.py:223-260` — `ManifestStore.save`: tempfile + flush + fsync(file) + os.replace + 親ディレクトリ fsync。**プロジェクト内で唯一 directory fsync を実施している箇所**
- `src/lorebook_chunker/analyzer.py:629-646` — `JapaneseAnalyzer.save` の atomic write
- `src/lorebook_chunker/ingest.py:602-624` — 現行 `_atomic_swap` (parent-dir fsync 欠如)
- `src/lorebook_chunker/ingest.py:313-317, 321-333` — staging ディレクトリ戦略 + entities/*.md の pre-populate (hardlink 不採用の理由コメント参照)

**実行結果の集約源**
- `src/lorebook_chunker/ingest.py:257-265` — `IngestResult` dataclass
- `src/lorebook_chunker/wiki.py:157-170` — `WikiStats` (token 合計、systemic_failure_aborted、preflight_calls)
- `src/lorebook_chunker/ner.py:37-43` — `AggregationStats` (total/accepted/skipped エンティティ数)
- `src/lorebook_chunker/chunker.py:29-35` — `ChunkerWarning` (kind/detail/char_offset)
- `src/lorebook_chunker/progress.py:1-138` — `ProgressReporter` (per-phase `time.perf_counter()` 内部計測済み、未 export)
- `src/lorebook_chunker/ingest.py:794-823` — 現行 `ingest_result.json` ライター (F-036)

**CLI 規約**
- `src/lorebook_chunker/cli.py:8-11` — `IDENTITY_BANNER` (新サブコマンドも流用)
- `--kebab-case` フラグ、`action="store_true"` boolean
- `argparse.RawDescriptionHelpFormatter` + 日本語 epilog

**テストパターン**
- `tests/conftest.py` — 現在空。共通 fixture 集約先
- `tests/test_ingest.py:17-89` — `_StubAnalyzer` / `_ScriptedLLM` (`tests/test_smoke.py:21` からクロスインポート)
- `tests/test_analyzer.py:14-29` — `requires_ginza` 可用性プローブパターン
- `tests/test_smoke.py:281,299` — `RUN_LLM=1` / `RUN_OLLAMA=1` gating
- `scripts/fast-ingest.sh:70-73` — 現在唯一の pre-CLI env validation

### Institutional Learnings

- `docs/solutions/` は現時点で存在しない。本計画のリリース後、atomic swap 契約・エラー taxonomy 設計は solutions 化候補
- prior plan `docs/plans/2026-04-21-001-feat-japanese-rag-chunking-plan.md` の Risks セクション (lines 883, 914) は「manifest の atomic write 必須」「single-writer 仮定」を明記済み。本計画の U6 はこの仮定を **directory レベル** に拡張する

### External References

- [clig.dev — Command Line Interface Guidelines](https://clig.dev/) — stdout/stderr split, exit code 規約
- [pip status_codes.py](https://github.com/pypa/pip/blob/main/src/pip/_internal/cli/status_codes.py) — 最小限の exit code テーブル (0/1/2/3/4/23) の reference
- [charset_normalizer (jawah)](https://github.com/jawah/charset_normalizer) — 2026 年時点の Python エンコーディング検出デファクト。`from_fp` による stream 対応
- [charset_normalizer Issue #121](https://github.com/jawah/charset_normalizer/issues/121) — CJK 短文での精度低下に関する作者コメント (→ `utf-8-sig` fallback 先行の根拠)
- [Python 3 `os.replace`](https://docs.python.org/3/library/os.html#os.replace) — POSIX 同一 FS 上での atomic 保証、Windows 上書き動作
- [alexwlchan — Atomic, cross-filesystem moves in Python](https://alexwlchan.net/2019/atomic-cross-filesystem-moves-in-python/) — EXDEV fallback の canonical パターン
- [Calvin Loncaric — How to Durably Write a File on POSIX](https://calvin.loncaric.us/articles/CreateFile.html) — parent-directory fsync の根拠
- [LWN — Exchanging two files (RENAME_EXCHANGE)](https://lwn.net/Articles/569134/) — Linux-only 三方向スワップ、portable fallback が優先される理由
- [Evan Jones — Durability: Linux File APIs](https://www.evanjones.ca/durability-filesystem.html) — macOS `F_FULLFSYNC` vs Linux `fsync` の差異 (本計画は F_FULLFSYNC 非採用)
- [ranx (AmenRa/ranx)](https://github.com/AmenRa/ranx) — 軽量な Python-native IR 評価ライブラリ。`recall@k / mrr@10 / ndcg@10` サポート、paired t-test 同梱
- [JQaRA (hotchpotch)](https://github.com/hotchpotch/JQaRA) — 将来統合候補。当面は qrels フォーマット互換性のみを意識

---

## Key Technical Decisions

- **エラー階層**: 単一基底 `LorebookError(Exception)` + per-category サブクラス + `exit_code: int` class attribute + `context: dict` instance attribute。既存 `AnalyzerVersionMismatchError` / `LLMError` 階層は `LorebookError` を継承させて統合 (後方互換は維持)。**理由**: pip / DVC の convention と整合、library 消費者が `except LorebookError` で一括捕捉可能、exit code map が CLI boundary の1箇所で完結
- **exit code 新規採番**: 既存 2/3/4/5/6/10 を後方互換維持。新規は 11-17 に配置 (`InputError=11, EncodingError=12, ChunkingError=13, TfidfError=14, EntityAggregationError=15, AtomicSwapError=16, RunReportError=17`)。**理由**: POSIX sysexits (64-78) は sendmail 由来で旧弊。2026 年デファクトは pip 方式の小テーブル + ドキュメント化
- **LLM 系 exit code 分離**: preflight 初期化失敗 (`_llm_factory` at `ingest.py:304-309`) は `LLMBackendUnavailableError(exit=3)`、実行時の wiki 生成失敗 (`wiki.generate_all` at `ingest.py:521-526`) は `WikiGenerationError(exit=6)`。既存の 3/6 契約を保持するため、`LLMPermanentError` を単一クラスにまとめず preflight/runtime を別クラスにラップする
- **doctor サブコマンドの exit code は ingest 系とは別空間**: `0` (pass) / `1` (warnings only) / `18` (env-critical failure) を採用。**理由**: `EncodingError=12` と `doctor` の critical failure を同一コードに載せると operator がコード分岐不能、`18` を確保して意味的独立を保つ。`lint` の 0/1/2 パターンは query 系 code との衝突があるため採用せず
- **エンコーディング検出**: `utf-8-sig` strict 先行 → fallback to `charset-normalizer` (soft dependency, `[full]` extra) → override `--encoding <name>`。**理由**: 2026 年の日本語 web は 99% UTF-8 (MDN/Wikipedia); Shift-JIS は legacy tail。soft-import により minimal インストールでも utf-8 ファイルは通る
- **charset-normalizer 閾値**: chaos < 0.3 かつ言語一致ヒットを要件とする。短ファイル (< 100 bytes) は検出不可として `EncodingError` で `--encoding` 明示を要求
- **atomic swap**: `os.replace` (not `os.rename`) + EXDEV explicit catch + copy-then-replace fallback + parent-dir fsync (best-effort; tmpfs での失敗は warning)。**`--no-atomic` は提供しない**。代わりに `--verify-swap` による post-swap SHA-256 照合を opt-in で提供。**理由**: atomic 保証を崩すフラグは安全性リスクが大きい; verification は付加機能として互換
- **durability 契約の honesty**: README に "同一 FS 上は atomic visibility、durability は clean shutdown 時点まで" を明記。`F_FULLFSYNC` (macOS) / 全ファイル fsync はパフォーマンス影響が大きく採用せず
- **run_report.json 導入 + ingest_result.json 削除**: 一度きりの破壊的変更。`schema_version: 1` で将来の互換性を担保。log.md (human-readable) は維持
- **schema v1 安定化の範囲**: キー名 (top-level と `phase_durations_seconds` 配下の6 phase 名 `analyzer_init`/`chunking`/`tfidf`/`ner`/`wiki`/`swap`) は **計画段階で fix**。exit code 番号、`exit_reason.context` の追加キー、`input.encoding_option` の値域などは **additive-only** (将来 v1 のまま拡張してよいが既存キー削除/リネーム/値削除は v2 必須)。これにより「実装時決定」と「v1 安定」の両立を明示化
- **phase timings は新規 instrumentation**: `ProgressReporter` は現状 `end()` で elapsed を print するだけで蓄積しない (`progress.py:78-94`)。U5 は (a) `ProgressReporter.__init__` に `self._durations: dict[str, float] = {}` を追加、(b) `end()` で elapsed を `_durations[name]` に記録してから state を reset、(c) `phase_timings() -> dict[str, float]` accessor を追加、(d) `IngestRunner.run` の 6 phase boundary を try/finally で囲んで `progress.start(name)` / `progress.end()` を呼び (失敗 phase も duration を残す)、を単位の**一連の変更**として scope する。既存の stderr 出力挙動は変えない
- **SkipReport 構造化**: `dataclass` with `reason: str` (enum-like 文字列: `"empty_file"`/`"encoding_decode_failed"`/`"encoding_detection_failed"`/`"permission_denied"`/`"file_not_found"`)。`IngestResult.warnings: list[str]` 内の既存 "empty file, skipped: ..." / "utf-8 decode failed, skipped: ..." 形式は tests で assertion されているため、log.md 生成時に同じ文字列を合成して後方互換を維持
- **ranx は optional**: `[bench]` extra 配下。CLI 実行には不要。ベンチマーク実行時のみ ImportError → 明確な install 指示メッセージ
- **qrels フォーマット**: JSONL、1行1クエリ、TREC 3列形式 (`qid docid grade`) 互換。`samples/expected.yaml` からの自動変換ツールを提供

---

## Open Questions

### Resolved During Planning

- **run_report.json vs ingest_result.json**: clean cut に決定。`ingest_result.json` は U5 で削除。Resolution: `v0.1.0` の段階で外部消費者実質不在、1リリース遅延した alias より1回の breaking change のほうがスキーマが安定化する
- **`--no-atomic` 実装**: 不採用。Resolution: atomic 保証を崩す opt-out は安全性負債が大きく、verification 側 (`--verify-swap`) のみ追加
- **RAG ベンチマーク gold set**: 自前 gold set + `ranx` を `[bench]` extra に。外部データセット統合は見送り。Resolution: スコープ最小化、`samples/expected.yaml` 起点の延長線
- **no-input-files の exit code / 例外**: `ConfigError(reason="no_input_files", input_dir=..., globs=[...])` を raise し **exit code 2 を維持**。`InputError(exit=11)` は per-file エラー (permission_denied 等) に予約。**理由**: 既存 `ingest.py:288-293` の exit 2 契約維持 + ConfigError が「設定が入力を生まなかった」の正しい意味論、`InputError` は「ファイル1件が処理不能だった」に限定したほうが run_report 読者の分岐が明確
- **exit code 11-17 の最終番号**: 本計画通り `InputError=11, EncodingError=12, ChunkingError=13, TfidfError=14, EntityAggregationError=15, AtomicSwapError=16, RunReportError=17`、`doctor` は別空間で `18` (critical)。既存 `cli.py:24-37` の lint (0/1/2) / query (0/2/3/4) と衝突しない (ingest 系 11-17 と doctor 系 18 はいずれも 10 以上で、lint/query は 4 未満)
- **`ProgressReporter` の 6 phase 名**: `analyzer_init`/`chunking`/`tfidf`/`ner`/`wiki`/`swap` を schema v1 の stable key として fix (上の Key Decisions 参照)。`IngestRunner.run` の対応コード箇所は `ingest.py:337` (analyzer_init), `ingest.py:402` (chunking), `ingest.py:429` (tfidf), `ingest.py:471` (ner), `ingest.py:491` (wiki), `ingest.py:544` (swap) — ここを try/finally wrap の対象とする
- **multi-inheritance 方針**: `AnalyzerVersionMismatchError` / `AnalyzerNEUnavailableError` / `ChunkFileCorruptError` は `LorebookError` **直接継承** (RuntimeError 多重継承は採用せず)。grep `except RuntimeError` で `schema.py` 側 3 クラスを捕捉する callers が無いことを U1 Verification で確認
- **schema v1 の context フィールド**: `exit_reason.context` の既知キー集合 = `{input_dir, globs, encoding_attempted, size_bytes, byte_offset, path, errno, detected_candidates}`。これらは v1 で stable。追加キーは additive-only (新しい error class が context kwarg を増やしても v1 互換)、既存キー削除/リネームは v2 必須

### Deferred to Implementation

- **charset-normalizer chaos 閾値の経験値**: 0.3 は docs 推奨 0.2 より緩め。実環境 fixture で tuning (schema に影響しない)
- **large corpus テストのファイル数**: 100 / 500 / 1000 のどこで時間閾値を置くかは実測後 (U8)
- **doctor サブコマンドが probe する Ollama モデル**: `llm-backend` オプション値に依存。実装時に backend=ollama だけ試行し anthropic は API key 存在のみ確認する方針を具体化
- **`doctor` の Ollama probe タイムアウト policy**: 5秒 hard timeout + `--skip-network` flag を U7 Files に追加するか実装時決定 (5秒タイムアウトのみで十分か、flag が必要かは real env で判断)
- **`.staging`/`.backup`/`.failed` 残存時の recover policy**: `doctor` が検出して案内まではするが、自動削除は行わない方針。ただし `.staging/` は既存の `ingest.py:315-317` rmtree 挙動を維持 (U8 test_resume_after_crash で文書化)

### Product-lens 戦略課題 (本計画では user が承認済み、将来の release で再検討)

これらは product-lens レビューで surfaced された戦略的指摘。本リリースでは user が「全7項目一括」を選択したため scope は維持するが、次 release 企画時に再訪する:

- **7項目一括 vs 分離リリース**: U1-U8 (operational hardening) と U9 (RAG eval) は消費者 audience が異なる。次 release では U1-U8 相当を単独で先行させる選択肢が残る
- **`--no-atomic` の underlying user need**: Issue #1 がなぜ `--no-atomic` を要求したかの原因は未検証。`--verify-swap` は tradeoff 方向が逆 (overhead 増)。実 user feedback で真の need が `--in-place` (debug 用 swap skip) だった場合、本計画の substitution は不十分
- **breaking change at public release タイミング**: `0de4f99 chore: prepare for public release` 直後の `ingest_result.json` 削除は早期 adopter 信頼を損ねる可能性。次 release で alias 層 1 cycle 維持 (`run_report.json` 生成 + `ingest_result.json` に `{"deprecated": true, "see": "run_report.json"}` の薄い stub を並存) を再検討余地あり
- **U9 self-built gold set の positioning**: 本計画は internal regression 用途。「RAG 検索評価ベンチマーク」という命名は外部比較を示唆しがちなため、README では「chunking params のチューニング用」と expectation setting を明記 (U9 Documentation Plan 側で対応)
- **`doctor` + `ingest --dry-run` の surface 重複**: 本計画では両方提供し、`--dry-run` が内部的に doctor を呼ぶ形で重複を抑制。次 release で user feedback に基づき、`doctor` 独立性を維持するか `--dry-run` に一本化するかを再判断

---

## High-Level Technical Design

> *これは想定アプローチの方向性を示す図であり、レビュー用のガイダンスです。実装エージェントは仕様ではなく context として扱ってください。*

### エラーフロー (raise → catch → report の経路)

```mermaid
flowchart TD
    subgraph domain["ドメインモジュール (raise)"]
        ingest_file["ingest._collect_input_files<br/>→ InputError"]
        encoding_layer["encoding.detect_encoding<br/>→ EncodingError"]
        analyzer_init["analyzer.build<br/>→ AnalyzerInitError"]
        chunker_call["chunker.chunk_document<br/>→ ChunkingError"]
        tfidf_call["tfidf.fit_transform<br/>→ TfidfError"]
        ner_agg["ner.aggregate_entities<br/>→ EntityAggregationError"]
        wiki_gen["wiki.generate_all<br/>→ WikiGenerationError<br/>(wraps LLMPermanentError)"]
        swap_op["_swap.atomic_swap<br/>→ AtomicSwapError"]
    end

    runner["IngestRunner.run<br/>(per-phase try/except)"]

    boundary["run_ingest<br/>(CLI boundary)"]

    subgraph outputs["出力"]
        exit_code["sys.exit(code)"]
        run_report["run_report.json<br/>(exit_reason.class,<br/>message, context)"]
        log_md["log.md<br/>(human-readable)"]
        skipped["skipped_files.jsonl<br/>(if skips > 0)"]
    end

    ingest_file --> runner
    encoding_layer --> runner
    analyzer_init --> runner
    chunker_call --> runner
    tfidf_call --> runner
    ner_agg --> runner
    wiki_gen --> runner
    swap_op --> runner

    runner --> boundary

    boundary -->|except LorebookError<br/>→ EXIT_CODE_MAP[type]| exit_code
    boundary -->|IngestResult.errors<br/>+ phase_timings| run_report
    boundary --> log_md
    boundary --> skipped
    boundary -->|except Exception<br/>→ exit 10 (unclassified)| exit_code
```

### `errors.py` モジュール形 (directional pseudo-code)

```python
# src/lorebook_chunker/errors.py — shape sketch, NOT implementation spec
class LorebookError(Exception):
    exit_code: int = 10
    def __init__(self, message: str, **context): ...
    def to_jsonable(self) -> dict: ...

class ConfigError(LorebookError): exit_code = 2
class LLMBackendUnavailableError(LorebookError): exit_code = 3   # preflight: LLM client init failed
class AnalyzerInitError(LorebookError): exit_code = 4
class InputError(LorebookError): exit_code = 11                   # non-empty glob but zero files → exit 2 variant (see U1 Approach)
class EncodingError(LorebookError): exit_code = 12
class ChunkingError(LorebookError): exit_code = 13
class TfidfError(LorebookError): exit_code = 14
class EntityAggregationError(LorebookError): exit_code = 15
class WikiGenerationError(LorebookError): exit_code = 6           # runtime: wiki.generate_all aborted
class AtomicSwapError(LorebookError): exit_code = 16
class RunReportError(LorebookError): exit_code = 17

EXIT_CODE_MAP: dict[type[LorebookError], int] = {
    cls: cls.exit_code for cls in (...)
}

def describe_exit_codes() -> str: ...  # shared by CLI epilog + run_report
```

### `run_report.json` v1 スキーマ (directional)

```jsonc
{
  "schema_version": 1,
  "lorebook_chunker_version": "0.2.0",
  "exit_code": 0,
  "exit_reason": null,               // or { "class": "InputError", "message": "...", "context": {...} }
  "started_at": "2026-04-23T05:15:00+09:00",
  "completed_at": "2026-04-23T05:16:12+09:00",
  "duration_seconds": 72.4,
  "phase_durations_seconds": {
    "analyzer_init": 3.1, "chunking": 12.4, "tfidf": 0.8,
    "ner": 8.2, "wiki": 45.1, "swap": 0.3
  },
  "input": {
    "input_dir": "samples",
    "recursive": false,
    "globs": ["*.txt"],
    "encoding_option": "auto",
    "files_processed": 4,
    "files_skipped": [
      { "path": "samples/bad.txt", "reason": "encoding_decode_failed",
        "detail": "utf-8 strict failed at byte 128", "encoding_attempted": "utf-8",
        "size_bytes": 4096 }
    ]
  },
  "output": {
    "output_dir": "out_optics",
    "chunks_generated": 127,
    "entities_generated": 34,
    "wiki_pages_written": 34
  },
  "analyzer": {
    "model_name": "ja_ginza_electra",
    "model_version": "5.2.0",
    "model_sha256": "e407d95f...",
    "spacy_version": "3.8.3",
    "ginza_version": "5.2.0",
    "sudachi_dict": "sudachidict_core 20250515"
  },
  "llm": {
    "backend": "anthropic",
    "model_id": "claude-opus-4-7",
    "total_input_tokens": 45231,
    "total_output_tokens": 8120
  },
  "warnings": ["chunker/soft_break: ..."]
}
```

---

## Implementation Units

- [ ] U1. **Error taxonomy + exit code テーブル + CLI boundary ハンドラ**

**Goal:** 単一基底 `LorebookError` とサブクラス階層を導入し、現在 `cli.py` / `ingest.py` で重複している exit code テーブルを1箇所に集約し、error context を機械可読化する。全ての後続 Unit が新しいエラー型を raise するための基盤。

**Requirements:** R5, R6

**Dependencies:** None (foundational)

**Files:**
- Create: `src/lorebook_chunker/errors.py`
- Create: `tests/test_errors.py`
- Modify: `src/lorebook_chunker/cli.py:13-37` (epilog 文字列を `errors.describe_exit_codes()` から生成)
- Modify: `src/lorebook_chunker/ingest.py:304-553` (各 `except Exception`/`raise ValueError` を適切な `LorebookError` 派生に置換)
- Modify: `src/lorebook_chunker/ingest.py:720-729` (`INGEST_EXIT_CODES` 削除、`errors.EXIT_CODE_MAP` を参照)
- Modify: `src/lorebook_chunker/ingest.py:549-553` (top-level `except Exception` → `except LorebookError` + fallback `except Exception`)
- Modify: `src/lorebook_chunker/schema.py:119-128` (`AnalyzerVersionMismatchError` / `AnalyzerNEUnavailableError` / `ChunkFileCorruptError` を `LorebookError` **直接継承** に変更。`RuntimeError` 多重継承は採用しない — Key Technical Decisions 参照)
- Modify: `src/lorebook_chunker/llm/__init__.py:28-41` (`LLMPermanentError` 系を `LorebookError` 派生化。**preflight/runtime 別クラス**: 初期化時の失敗を `LLMBackendUnavailableError(exit=3)`、wiki 実行時の失敗を `WikiGenerationError(exit=6)` にマップして既存 exit 3/6 契約を保持)

**Approach:**
- `LorebookError` は `exit_code: int` class attribute + `context: dict[str, str | int | float]` instance attribute + `to_jsonable() -> dict` メソッドを持つ。`__init__(message, **context)` で context kwargs を受ける
- 既存 exit code (2/3/4/5/6/10) は per-class `exit_code` に保存して後方互換維持。新規 (11-17) を追加
- `ingest.py:288-293` の "no input files" → `ConfigError(reason="no_input_files", input_dir=str(path), globs=[...])` (exit 2 維持)。**per-file エラー (permission_denied / file_not_found)** は `InputError(exit=11)` を使用 — 「設定が入力を生まなかった」と「1件ずつのエラー」を別クラスで区別
- `ingest.py:304-309` の preflight LLM 失敗 → `LLMBackendUnavailableError(exit=3, backend=..., reason=...)` でラップ (既存 exit 3 契約を preflight 専用に維持)
- `ingest.py:340-343` の analyzer init 失敗 → `AnalyzerInitError(...)` でラップ
- `ingest.py:424-427` の "no chunks" → `ChunkingError(reason="zero_chunks")`
- `ingest.py:521-526` の wiki runtime 失敗 → `WikiGenerationError(exit=6, ...)` でラップ (既存 exit 6 契約を runtime 専用に維持)
- `ingest.py:602-624` の swap 失敗 → `AtomicSwapError(source=..., target=..., errno=...)` (U6 で詳細実装)
- CLI boundary (`run_ingest`) で `except LorebookError as e: return e.exit_code`; `IngestResult.errors` に `[{"class": e.__class__.__name__, "message": str(e), "context": e.context}]` を記録

**Execution note:** 先に characterization テストで現在の exit code 挙動 (2/3/4/5/6/10) を lock してから rewrite。これにより後方互換の silent breakage を防ぐ。

**Patterns to follow:**
- `src/lorebook_chunker/llm/__init__.py:28-41` の sub-exception 階層
- `src/lorebook_chunker/schema.py:119-128` の custom exception 命名 (`*Error`、RuntimeError 派生)
- `src/lorebook_chunker/lint.py:34-36` の 3-tier exit code パターン (0/1/2)
- JSON 出力規約: `json.dumps(..., ensure_ascii=False)`

**Test scenarios:**
- Happy path: 正常 ingest が引き続き `exit_code=0` を返す (characterization)
- Happy path: `errors.describe_exit_codes()` が CLI epilog と一致する文字列を返す
- Happy path: 各 `LorebookError` サブクラスが `to_jsonable()` で `{"class", "message", "context"}` を round-trip
- Edge case: context に non-serializable 値を渡すと `to_jsonable()` は str 変換で degrade
- Error path: no-input-files で `ConfigError(reason="no_input_files")` → exit 2 (既存契約)、`result.errors[0].class == "ConfigError"`
- Error path: per-file `PermissionError` → `InputError(exit=11)` にラップ → exit 11、`result.errors[0].class == "InputError"`
- Error path: preflight LLM 失敗で `LLMBackendUnavailableError(exit=3)` → exit 3 (既存契約)
- Error path: runtime wiki 失敗で `WikiGenerationError(exit=6)` → exit 6 (既存契約)
- Error path: `AtomicSwapError(reason="EXDEV")` が内側の `_atomic_swap` から伝播 → exit 16、outer `except Exception` に捕まらない
- Error path: 未分類の `ValueError` が CLI boundary に到達 → fallback `except Exception` で exit 10、`result.errors[0].class == "Exception"` (不明扱い)
- Integration: `tests/test_cli_smoke.py` / `tests/test_ingest.py` の既存 exit-code assertion (特に exit 2/3/6) が全て通過する
- Integration: `src/lorebook_chunker/cli.py:13-37` の epilog と `errors.EXIT_CODE_MAP` が同一 source から導出 (test が両方 parse して照合)
- Integration: `grep -rn 'except RuntimeError' src/ tests/` で `schema.py` の 3 クラスを捕捉する caller が無いことを確認 (multi-inheritance 不要性の根拠)

**Verification:**
- `pytest tests/test_errors.py tests/test_ingest.py tests/test_cli_smoke.py` 全 pass
- `lorebook-chunker ingest --help` の epilog に新コード 11-17 が表示
- `grep -rn INGEST_EXIT_CODES src/` は `errors.py` のみヒット

---

- [ ] U2. **入力探索オプション (`--recursive` / `--glob`)**

**Goal:** `glob("*.txt")` トップレベル固定を撤廃し、再帰探索と任意パターン指定を可能にする。**あわせて `tests/conftest.py` の stub 集約 (U8 から前倒し)** を本 Unit で実施し、以降の U2-U6 テストが consolidated fixture 上で書ける状態にする。

**Requirements:** R1, R2

**Dependencies:** U1 (`InputError` / `ConfigError` を raise)

**Files:**
- Modify: `src/lorebook_chunker/cli.py:48-49` (`--recursive/-r` store_true、`--glob` 文字列 default `"*.txt"`)
- Modify: `src/lorebook_chunker/ingest.py` (`IngestConfig` に `recursive: bool = False`, `globs: tuple[str, ...] = ("*.txt",)`)
- Modify: `src/lorebook_chunker/ingest.py:564-567` (`_collect_input_files` rewrite)
- Create: `tests/test_input_handling.py`
- Modify: `tests/conftest.py` (U8 から前倒し: `_StubAnalyzer` / `_ScriptedLLM` / `_ScriptedLLMClient` を移動、`ingest_config_factory(tmp_path, **overrides) -> IngestConfig` と `tmp_samples_dir(tmp_path, content_map)` fixture を追加)
- Modify: `tests/test_ingest.py` / `tests/test_smoke.py` / `tests/test_wiki.py` (stub の local 定義を削除、conftest fixture 経由の import に)

**Approach:**
- `--glob` はカンマ区切り許容: `"*.txt,*.md"` → `("*.txt", "*.md")`。internally dedupe after resolve
- `_collect_input_files`: recursive なら `rglob`、それ以外は `glob`。各 pattern iterate → `set` で resolved path dedupe → sort (安定順序)
- 絶対パスパターン (`/etc/**`) や path separator を含むものは `ConfigError`
- 0 ファイル時は U1 の `InputError` を raise (現行 exit 2 挙動を維持)

**Patterns to follow:**
- `--skip-wiki` boolean flag (`cli.py:50`)
- `IngestConfig` dataclass (`ingest.py` 上部の `@dataclass`)
- `argparse.RawDescriptionHelpFormatter` + 日本語 help (`cli.py:43-115`)

**Test scenarios:**
- Happy path: デフォルト (flag 無し) で top-level `*.txt` のみ検出 (regression)
- Happy path: `--recursive` 平坦 dir → top-level と同じファイル集合
- Happy path: `--recursive` ネスト dir → `subdir/a.txt` や `subdir/deep/b.txt` も検出
- Happy path: `--glob "*.md"` → `.txt` は無視、`.md` のみ
- Happy path: `--glob "*.txt,*.md"` → 両方、sort 安定、重複なし
- Edge case: `--glob "*.txt"` で 0 件 → `InputError` (exit 2)
- Edge case: 同一ファイルが複数 glob にマッチ → dedup by resolved path
- Edge case: ネスト 5 階層 / 1000 ファイル → stack overflow せず完走
- Error path: `--glob "/absolute/pattern"` → `ConfigError`
- Error path: `--glob ""` (空文字) → `ConfigError`
- Integration: `ingest --help` に両オプションが日本語説明付きで表示

**Verification:**
- `lorebook-chunker ingest -r --glob "*.txt,*.md" tests/fixtures/samples out_tmp/` が nested dir を横断して走る
- 既存 `tests/test_ingest.py` が変更無しで通過

---

- [ ] U3. **エンコーディング検出パイプライン**

**Goal:** `read_text(encoding="utf-8")` 一本を止め、`utf-8-sig` strict → `charset-normalizer` soft → explicit `--encoding` の3段構えに。

**Requirements:** R3

**Dependencies:** U1 (`EncodingError` / `ConfigError`)

**Files:**
- Create: `src/lorebook_chunker/encoding.py`
- Create: `tests/test_encoding.py`
- Create: `tests/fixtures/encoding_samples/` (Shift-JIS / CP932 / UTF-8 BOM / EUC-JP / invalid bytes fixture)
- Modify: `src/lorebook_chunker/ingest.py:353-368` (`read_text` 呼び出しを `encoding.read_text_with_encoding` に置換)
- Modify: `src/lorebook_chunker/ingest.py` (`IngestConfig.encoding: str = "auto"`)
- Modify: `src/lorebook_chunker/cli.py` (`--encoding` string、`"auto"` default; 値のバリデーションは遅延)
- Modify: `pyproject.toml:42-45` (`[project.optional-dependencies]` に `full = ["charset-normalizer>=3.4"]` を追加)

**Approach:**
- `encoding.read_text_with_encoding(path: Path, *, encoding: str) -> tuple[str, str]` — (text, actual_encoding) を返す
- 処理順:
  1. `encoding != "auto"` なら `path.read_text(encoding=encoding)` を strict で実行。失敗時 `EncodingError(path, encoding, byte_offset)`
  2. `encoding == "auto"` なら `utf-8-sig` strict を試す (BOM 有りも無しも処理、`﻿` 自動除去)
  3. 失敗時、`charset-normalizer` を try-import。未導入なら `EncodingError(hint="install charset-normalizer via [full] extra or pass --encoding")`
  4. `charset_normalizer.from_path(path, steps=5, chunk_size=512)` → top match の `chaos < 0.3` かつ language hit があれば採用。それ以外は `EncodingError(path, detected_candidates=[...], hint="ambiguous, pass --encoding explicitly")`
- ファイルサイズ < 100 bytes の短文 `auto` は検出せず `utf-8-sig` のみ試行 → 失敗なら `EncodingError(reason="too_short_to_detect")`
- `EncodingError` の context には `tried: list[str]`, `size_bytes: int`, `byte_offset: int | None`, `detected_candidates: list[{encoding, chaos, confidence}]`

**Patterns to follow:**
- Soft-import: `tests/test_analyzer.py:14-23` の `_ginza_available()`
- エラーメッセージに英語 substring を残す (tests が assertion する): `"utf-8 decode failed"` 等は新実装でも同文字列を含める

**Test scenarios:**
- Happy path: plain UTF-8 → `("text", "utf-8-sig")` (BOM 有無同一 API)
- Happy path: UTF-8 with BOM → 本文に `﻿` が残らない、`actual_encoding == "utf-8-sig"`
- Happy path: `--encoding utf-8` 明示 → detector 呼ばれない、`actual_encoding == "utf-8"`
- Edge case: `--encoding auto` + charset-normalizer 導入 + Shift-JIS → `("text", "shift_jis")`
- Edge case: `--encoding auto` + charset-normalizer 未導入 + utf-8-sig 不通 → `EncodingError` に install hint
- Edge case: `--encoding auto` + CP932 の半角カナ → 成功
- Edge case: 50 bytes の短いファイルが auto → `EncodingError(reason="too_short_to_detect")`
- Edge case: chaos = 0.5 の曖昧ファイル → `EncodingError(detected_candidates=[...])`
- Error path: `--encoding foo` (無効名) → `ConfigError`
- Error path: utf-8 宣言で非 utf-8 byte 含む → `EncodingError(byte_offset=<int>)`
- Integration: 実 ingest 経由で Shift-JIS ファイルが chunks.jsonl に正しい日本語として出る
- Integration: `actual_encoding` が (後続 U4/U5 で) skip report / run_report に記録できる形で返る
- Regression: `tests/test_ingest.py:229-233` の既存 fixture (bad-UTF-8 bytes) を `--encoding utf-8` 明示で呼ぶ形に書き換え → warning 文字列 `"utf-8 decode failed"` が引き続き assertion される。**理由**: `--encoding auto` default で charset-normalizer 導入時 (`[full]` extra) に bad-UTF-8 が Shift-JIS として detect 成功すると warning 無し path に入り既存 assertion が破綻する。test fixture を explicit encoding で固定して CI 環境差 ([full] 有無) に依存しない

**Verification:**
- `pytest tests/test_encoding.py` が pass
- `tests/test_ingest.py:209-233` (既存の空 / 壊れ UTF-8) が **`--encoding utf-8` 明示で呼ぶ形に書き換えて** 通過
- `pip install -e '.[full]'` で charset-normalizer が入る
- `[full]` extra 導入環境と未導入環境の両方で `pytest tests/test_ingest.py` が通る (CI matrix or 手元で検証)

---

- [ ] U4. **SkipReport + `skipped_files.jsonl`**

**Goal:** 現在 flat 文字列警告として捨てているスキップ理由を構造化し、`skipped_files.jsonl` として機械可読に出力する。

**Requirements:** R4

**Dependencies:** U1 (reason 型定義), U2 (recursive 時の path), U3 (encoding 失敗情報)

**Files:**
- Modify: `src/lorebook_chunker/schema.py` (`SkipReport` dataclass を追加)
- Modify: `src/lorebook_chunker/ingest.py:257-265` (`IngestResult` に `skips: list[SkipReport]` を追加、`skipped_files: int` は `@property` で `len(self.skips)` として後方互換維持)
- Modify: `src/lorebook_chunker/ingest.py:353-368` (warning 文字列 append を `self.skips.append(SkipReport(...))` に置換、ただし log.md 用に warnings にも同内容の文字列を残す)
- Modify: `src/lorebook_chunker/ingest.py:484-489` (staging 書き出し箇所で `skipped_files.jsonl` を emit、`len(skips) > 0` のときのみ)
- Create: `tests/test_skip_report.py`

**Approach:**
- `SkipReport`:
  ```python
  @dataclass
  class SkipReport:
      path: str                  # resolved absolute path
      reason: str                # "empty_file" | "encoding_decode_failed" | "encoding_detection_failed" | "permission_denied" | "file_not_found"
      detail: str | None = None  # human-readable, may be None
      encoding_attempted: str | None = None
      size_bytes: int | None = None
  ```
- JSONL 1行1ファイル、`ensure_ascii=False`
- `log.md` への文字列 append は維持 (既存 test assertion `"empty file, skipped: ..."` / `"utf-8 decode failed, skipped: ..."` の互換性)
- Staging ディレクトリ内に書き出し → U6 の atomic swap と一緒に公開
- Skip 0件時は `skipped_files.jsonl` を作成しない (出力 dir を clean に保つ)

**Patterns to follow:**
- `ChunkerWarning` (`chunker.py:29-35`) の dataclass shape
- JSONL emission: `wiki.py` 内の manifest 書き出しスタイル
- Staging write 手順 (`ingest.py:484-489`)

**Test scenarios:**
- Happy path: skip 0件 → `skipped_files.jsonl` 作成されない
- Happy path: 空ファイル1件 + bad-UTF-8 1件 + OK 1件 → JSONL 2行、OK は chunks.jsonl に
- Happy path: `IngestResult.skipped_files` (互換 property) が `len(skips)` と一致
- Edge case: reason 値が enum-like 文字列として固定 (snapshot テスト)
- Edge case: recursive 探索経由の skip → path は絶対
- Edge case: 同ファイルが複数 reason で skip されない (一度 skip されたら後続 phase に渡さない)
- Integration: 既存の `test_ingest.py:209-233` が `IngestResult.skips` 参照の assertion に書き換えられる (warnings 文字列 assertion は log.md 互換部分のみに縮約)
- Integration: `skipped_files.jsonl` の行数 == `run_report.json["input"]["files_skipped"]` の長さ

**Verification:**
- JSONL ファイルが `jq -c . skipped_files.jsonl | wc -l` で行数正しい
- `SkipReport.reason` の有効値が `schema.py` のコメントで列挙される

---

- [ ] U5. **Phase timings + `run_report.json` (ingest_result.json 置換)**

**Goal:** 実行レポートの正規化。phase duration、entities_generated、analyzer/LLM モデルメタ、skip 記録、exit reason を1つの `run_report.json` に集約する。成功・失敗の両方で emit。

**Requirements:** R9

**Dependencies:** U1 (exit_reason 構造), U4 (SkipReport 記録)

**Files:**
- Create: `src/lorebook_chunker/run_report.py`
- Create: `tests/test_run_report.py`
- Modify: `src/lorebook_chunker/progress.py:1-138` — (a) `__init__` に `self._durations: dict[str, float] = {}` を追加、(b) 既存 `end()` で elapsed を `self._durations[name]` に記録してから reset、(c) `phase_timings() -> dict[str, float]` accessor を追加
- Modify: `src/lorebook_chunker/ingest.py` (`IngestRunner.run` の 6 phase boundary に `progress.start(name)` + try/finally で `progress.end()` を wrap。対象行: `ingest.py:337` (analyzer_init), `ingest.py:402` (chunking), `ingest.py:429` (tfidf), `ingest.py:471` (ner), `ingest.py:491` (wiki), `ingest.py:544` (swap))
- Modify: `src/lorebook_chunker/ingest.py:257-265` (`IngestResult` に `phase_timings: dict[str, float]` と `entities_generated: int` を追加)
- Modify: `src/lorebook_chunker/ingest.py:794-823` (`ingest_result.json` 書き出しを `run_report.json` 書き出しに **置換** — 旧コード削除)
- Modify: `README.md` — (a) 新セクション「実行レポート (run_report.json)」追加、(b) Breaking Change 告知、(c) **line 25/47/324/375 の `ingest_result.json` 言及を `run_report.json` に書き換え**、(d) line 47 mermaid ダイアグラムの `ingest_result.json` ノードをリネーム

**Approach:**
- `run_report.py`:
  - `@dataclass class RunReport` with `to_json_dict() -> dict` method
  - `schema_version: int = 1` を class 定数
  - `write(path: Path, report: RunReport)` helper (atomic write: tempfile + fsync + replace + parent-dir fsync)
- **`ProgressReporter` の timing accumulation は新規 instrumentation** (現状 `end()` は elapsed を print するだけで蓄積しない、Key Technical Decisions「phase timings は新規 instrumentation」を参照):
  - `__init__` に `self._durations: dict[str, float] = {}` を追加
  - `end()` 内で `elapsed = time.perf_counter() - self._phase_start` を計算した後、`self._durations[self._phase] = elapsed` を実行してから state reset
  - 新規 `phase_timings() -> dict[str, float]`: `dict(self._durations)` の snapshot を返す
- **IngestRunner.run の 6 phase を明示 try/finally wrap**:
  - `progress.start("analyzer_init")` → analyzer init (`ingest.py:337`) → `progress.end()`
  - 同様に `chunking` (`ingest.py:402`), `tfidf` (`ingest.py:429`), `ner` (`ingest.py:471`), `wiki` (`ingest.py:491`), `swap` (`ingest.py:544`)
  - **失敗 phase も duration を記録** (try/finally で `end()` を finally 節に置く → 例外伝播しても `_durations[name]` が残る)
  - phase 名は schema v1 の stable key として fix (Key Decisions 参照)
- 最終段で `result.phase_timings = progress.phase_timings()`
- `entities_generated` は `AggregationStats.accepted_entities` を `IngestResult` に持ち上げ (現在 log.md 経由で消えている)
- Writer は `exit_code == 0` か否かに関わらず常に emit。既存 stdout `--format json` 出力は `run_report` の `to_json_dict()` を直接使う

**Execution note:** Schema contract test を先に書く (schema_version + 必須キー全てを assert)。v1 確定は U5 完了時点、以降の変更は `schema_version` bump を伴う。

**Patterns to follow:**
- `WikiStats` (`wiki.py:157-170`) の shape と命名
- `AggregationStats` (`ner.py:37-43`)
- Atomic write: `wiki.py:223-260` (`ManifestStore.save`)
- `json.dumps(..., ensure_ascii=False, indent=2)`

**Test scenarios:**
- Happy path: 正常 ingest → `run_report.json` に必須キー全部、`exit_code: 0`, `exit_reason: null`
- Happy path: `phase_durations_seconds` の各値が 0 以上、総和 ≒ `duration_seconds` (±10%)
- Happy path: `phase_durations_seconds` が 6 phase 全部含む (`analyzer_init`/`chunking`/`tfidf`/`ner`/`wiki`/`swap`) — 固定 key set を snapshot で lock
- Happy path: `schema_version == 1` を固定値として lock
- Happy path: `analyzer.model_sha256` が `analyzer.json` の対応キーと一致
- Edge case: exit 6 (wiki permanent error) → `run_report.json` 出力、`exit_reason.class == "WikiGenerationError"`、**`phase_durations_seconds.wiki` は失敗までの経過時間を持つ** (try/finally により)
- Edge case: 全ファイル skip → `chunks_generated: 0`, `entities_generated: 0`, report は出力
- Edge case: `entities_generated` は unique entity 数 (登場回数ではない)
- Edge case: phase 途中で例外発生 → `_durations[name]` に失敗時点までの経過を記録 (`try/finally` の finally が走る)
- Error path: `run_report.json` 書き込み不可 (disk full simulation) → `RunReportError`、exit 17
- Integration: `ingest_result.json` は **作成されない** (README/docs 含めた広域 grep が空)
- Integration: 既存 stdout `--format json` 出力が `run_report.to_json_dict()` と一致
- Integration: U4 で emit された `skipped_files.jsonl` の行数 == `run_report.json["input"]["files_skipped"]` 長さ

**Verification:**
- `jq .schema_version out/run_report.json` が `1`
- `jq '.phase_durations_seconds | keys | sort'` が `["analyzer_init","chunking","ner","swap","tfidf","wiki"]` を返す
- `grep -rn ingest_result.json src/ tests/ docs/ README.md scripts/` 空 (既知の埋込箇所: `README.md` line 25/47/324/375 の mermaid ノード含む文字列を書き換え済み、`src/lorebook_chunker.egg-info/PKG-INFO` は build 時再生成のため対象外)
- `tests/test_run_report.py` の snapshot test が full dict 構造を lock

---

- [ ] U6. **Atomic swap ハードニング (`os.replace` + EXDEV + `--verify-swap` + parent-dir fsync + 契約文書化)**

**Goal:** directory swap の durability と cross-device 挙動を明確化。契約を README に明記。SHA-256 照合による post-swap verification を opt-in で提供。

**Requirements:** R7, R8 (verify-swap 採用、`--no-atomic` は不採用)

**Dependencies:** U1 (`AtomicSwapError`)

**Files:**
- Create: `src/lorebook_chunker/_swap.py` (swap 処理 + manifest SHA-256 生成/照合を抽出)
- Create: `tests/test_atomic_swap_hardening.py`
- Modify: `src/lorebook_chunker/ingest.py:602-624` (`_atomic_swap` を `_swap.atomic_swap` 呼び出しに置換、旧コード削除)
- Modify: `src/lorebook_chunker/ingest.py` (`IngestConfig` に `verify_swap: bool = False`)
- Modify: `src/lorebook_chunker/cli.py` (`--verify-swap` store_true フラグ追加)
- Modify: `README.md:480` 付近 (新セクション「Atomic swap contract」)

**Approach:**
- `_swap.atomic_swap(target: Path, staging: Path, *, verify: bool) -> None`:
  1. `target.parent.mkdir(parents=True, exist_ok=True)`
  2. `verify=True` なら staging 内の全ファイルを walk → `(relpath, size, sha256)` tuple を `staging/.swap.manifest.sha256` に書き出し (1行1ファイル)
  3. 既存 target があれば `target.name + ".backup"` に rename
  4. `os.replace(staging, target)` を試行
     - `OSError` (errno `EXDEV`) → `shutil.copytree(staging, target.with_name(target.name + ".swap-tmp"))` → `os.replace(<tmp>, target)` → `shutil.rmtree(staging)`
     - `OSError` その他 → `AtomicSwapError(reason=..., errno=..., source=..., target=...)`
  5. 親ディレクトリ fsync (best-effort):
     ```python
     try:
         dir_fd = os.open(target.parent, os.O_DIRECTORY | os.O_RDONLY)
         os.fsync(dir_fd)
     except OSError:
         logger.warning("parent-directory fsync not supported, continuing")
     finally:
         try: os.close(dir_fd)
         except Exception: pass
     ```
  6. `verify=True` なら target を walk して sha256 を再計算、manifest と比較 → mismatch で `AtomicSwapError(reason="verify_mismatch", path=...)` (この場合 backup は残す)
  7. verify 成功時は `target/.swap.manifest.sha256` を削除 (出力 dir を clean に保つ)
  8. backup を `shutil.rmtree` (clean 終了時のみ)
- `--no-atomic` は **提供しない**
- **`--verify-swap` の意味論**: 同一 FS 上の `os.replace` は directory-entry のメタデータ操作であり bytes を再読しないため、staging と target は同一 inode を指す。したがって pre/post hash は **必ず一致** し、検出価値は主に (a) EXDEV fallback 経路で `shutil.copytree` がバイトコピーを行うため破損検知、(b) I/O 層の稀な bit-flip / FS 破損の発見、に限定される。**同一 FS 上の opt-in にも意味があるのは**、破損検知より「swap 完了までコピー整合性を明示確認したい」という運用契約ニーズ (例: regulated pipeline) — README に明記して過剰期待を防ぐ
- README セクション例:
  > **Atomic swap contract**
  > - 同一ファイルシステム上では、出力ディレクトリの切替は atomic (リーダーは旧完全版 or 新完全版のみを観測、中間状態は観測不能)
  > - クロスデバイス境界では `<output>.swap-tmp` に `copytree` → `os.replace` にフォールバック。コピー中の一時状態は sibling dir として存在
  > - 電源断耐性 (durability under power loss) は保証しない。クリーンシャットダウン時点まで (`fsync` + parent-dir fsync 実施)
  > - 失敗時の sibling dir: `<output>.staging/`, `<output>.backup/`, `<output>.failed/`, `<output>.swap-tmp/` — 再実行前に削除または保全
  > - `--verify-swap` で post-swap SHA-256 照合を実施 (opt-in、数秒〜数十秒のオーバーヘッド)。**同一 FS 上では `os.replace` が inode 操作のため pre/post hash は構造的に一致する — 本 flag の検出価値は主にクロスデバイス (EXDEV) fallback 経路および I/O 層の稀な破損**

**Patterns to follow:**
- `wiki.py:223-260` の atomic write + parent-dir fsync
- `analyzer.py:629-646` の tempfile + fsync + replace
- 既存 staging layout (`ingest.py:313-317, 321-333`)

**Test scenarios:**
- Happy path: same-fs swap → `os.replace` ルート、target 置換、backup 削除
- Happy path: `--verify-swap` で clean copy → manifest 一致、swap 成功、`.swap.manifest.sha256` は verify 成功後に削除される
- Happy path: `target` 不在 (初回 run) → backup 段階スキップ、新 target 作成
- Edge case: parent-dir fsync が tmpfs で失敗 → warning ログ、swap 継続
- Edge case: `--verify-swap` 無しだと manifest 生成されない
- Edge case: `--verify-swap` on same-fs → 構造的に hash 一致 (`os.replace` は inode 操作) を assert。破損検知能力は限定的でも code path は正しく通る
- Edge case: `--verify-swap` on EXDEV path → `copytree` 経路で bytes が実際にコピーされるため mismatch 検知が意味を持つ (simulation で破損を注入して assert)
- Error path: `monkeypatch` で `os.replace` を `OSError(errno=errno.EXDEV)` 発生 → EXDEV branch 通過、`shutil.copytree` + `os.replace` 完走
- Error path: `monkeypatch` で `os.replace` を `OSError(errno=errno.EACCES)` 発生 → `AtomicSwapError`、exit 16、staging 保全 (手動調査可能)
- Error path: `--verify-swap` 中に EXDEV path で target ファイルが mutate された simulation → `AtomicSwapError(reason="verify_mismatch")`、backup 保全
- Error path: backup rename 後に `os.replace(staging, target)` が落ちた場合 → target 不在 + backup あり → `AtomicSwapError` で restart 可能な状態
- Integration: 既存 `tests/test_ingest.py:260-280` (`test_ingest_atomic_swap_replaces_existing`) が無修正で通過
- Integration: 既存 `tests/test_smoke.py:196-236` (`test_ingest_preserves_existing_output_on_failure`) が無修正で通過
- Integration: `--verify-swap` が `run_report.json.phase_durations_seconds.swap` に計測可能な delta を追加

**Verification:**
- `README.md` に「Atomic swap contract」セクション追加
- `rg --files-with-matches 'os\.rename' src/lorebook_chunker/` が `_swap.py` のみ (EXDEV fallback パス) または 0 件
- EXDEV テストが simulate mode で pass (実際の cross-device mount 不要)

---

- [ ] U7. **`--dry-run` / `doctor` サブコマンド + 依存プロファイル文書化**

**Goal:** operator が「この環境で本当に走るか」を数秒で確認できる手段を提供。インストールプロファイル (minimal / fast / full / bench / dev) を README で明示。

**Requirements:** R10, R11

**Dependencies:** U1 (環境チェック失敗の型)

**Files:**
- Create: `src/lorebook_chunker/doctor.py`
- Create: `tests/test_doctor.py`
- Modify: `src/lorebook_chunker/cli.py` (`doctor` サブコマンド追加 + `ingest` に `--dry-run` フラグ追加)
- Modify: `pyproject.toml:36-45` (`[full]`, `[bench]` extras 追加 + `dev`/`fast` コメント整理)
- Modify: `README.md:241-307` (インストールセクション拡張、プロファイルテーブル)

**Approach:**
- `doctor.py` はチェックリスト実行:
  - Python version in `>=3.11,<3.13` → fail で exit 18
  - `spacy` / `ginza` import + バージョン範囲
  - `ja_ginza_electra` or `ja_ginza` (fast mode) モデル load 可否 (後者は `fast` extra)
  - `charset-normalizer` import (warning only)
  - `anthropic` client import + `ANTHROPIC_API_KEY` env (backend=anthropic 時)
  - `ollama` client import + `ollama.show(tag)` 成功 (backend=ollama 時)
  - `ranx` import (`[bench]` 要請時のみ、warning only)
  - 出力ディレクトリの親が書き込み可
- Exit codes: **doctor は ingest 系 (11-17) と別空間を使用**。`0` (pass) / `1` (warnings only) / `18` (env-critical failure)。**理由**: `EncodingError=12` と衝突させないため、かつ `lint` の 0/1/2 パターンは query 系 code と重複するため 18 を確保
- `doctor --json` で機械可読出力
- `ingest --dry-run`: doctor を先に走らせ、入力探索 (recursive / glob 適用) のみ実施して `files_discovered`, `first_file_encoding_probe` を stdout に JSON で返す。`output_dir` は作成しない
- README プロファイルテーブル:
  | profile | command | 用途 |
  |---|---|---|
  | minimal | `pip install lorebook-chunker` | 最小、ELECTRA 版 GiNZA、UTF-8 のみ |
  | fast | `pip install lorebook-chunker[fast]` | ja_ginza (非 transformer) で CPU 5-10x 高速、NER 精度若干低下 |
  | full | `pip install lorebook-chunker[full]` | + charset-normalizer (エンコーディング自動検出) |
  | bench | `pip install lorebook-chunker[bench]` | + ranx (RAG 検索品質ベンチマーク) |
  | dev | `pip install lorebook-chunker[dev]` | + pytest (コントリビュータ向け) |

**Patterns to follow:**
- `lint` サブコマンド構造 (`cli.py:24-29`)
- `requires_ginza` probe (`tests/test_analyzer.py:14-23`)
- `scripts/fast-ingest.sh:70-73` の env validation
- `IDENTITY_BANNER` (`cli.py:8-11`)

**Test scenarios:**
- Happy path: clean env で `doctor` → exit 0
- Happy path: `doctor` で `ANTHROPIC_API_KEY` 未設定 + backend=anthropic → exit 18、具体的ヒント
- Happy path: `doctor` で charset-normalizer 未導入 → exit 1 (warning)、install hint
- Happy path: `doctor --json` が全チェック結果を構造化 JSON で返す (`{checks: [{name, status, detail}], summary: {passed, warnings, failures}}`)
- Edge case: `ingest --dry-run` with valid input → exit 0、`out/` 未作成、stdout JSON
- Edge case: `ingest --dry-run` with 0 files → exit 2 (`ConfigError` の no_input_files、Open Questions → Resolved 参照)
- Edge case: `--dry-run` が最初のファイルの 64 KiB sample で encoding probe を実施
- Edge case: `doctor` exit code (0/1/18) が ingest 系 (2-17) と構造的に衝突しない — test が `errors.EXIT_CODE_MAP.values()` と doctor の code 集合で intersection を assert
- Error path: Python 3.10 env → `doctor` exit 18
- Error path: `lorebook-chunker doctor invalid-arg` → `ConfigError`
- Integration: `test_doctor.py` が `subprocess` 経由で呼ぶ (CLI 形態テスト)
- Integration: README に 5 profile 全部載っている (grep テスト)

**Verification:**
- `lorebook-chunker doctor` clean env で exit 0
- `lorebook-chunker ingest --dry-run samples/ out/` が `out/` 作らず exit 0
- `pip install -e '.[full]'` / `pip install -e '.[bench]'` が成功
- README の「インストール」セクションに 5 プロファイルテーブル

---

- [ ] U8. **統合テストスイート (encoding variants / atomic swap failure / resume / large corpus)**

**Goal:** production-readiness 側の回帰を防ぐ統合テストを追加。`conftest.py` の stub 集約は **U2 の先行作業として前倒し** (下記 Dependencies 参照)、本 Unit は残りの cross-cutting scenario に focus。

**Requirements:** R12

**Dependencies:** U2, U3, U4, U5, U6 (これらの新挙動をテスト対象)。`conftest.py` stub 集約は **U2 に前倒し** — U2-U6 は最初から consolidated fixture を前提にテストを書くため、本 Unit でリファクタを遅延させない

**Files:**
- Create: `tests/fixtures/encoding_samples/` (Shift-JIS / CP932 / UTF-8-BOM / EUC-JP / invalid-bytes)
- Create: `tests/fixtures/large_corpus_generator.py` (100-500 files 生成)
- Create: `tests/test_resume_after_crash.py`
- Create: `tests/test_large_corpus.py` (env-gated)
- (U2 で `tests/conftest.py` stub 集約 + `tests/test_ingest.py` / `test_smoke.py` / `test_wiki.py` の import 書き換えが完了している前提)

**Approach:**
- `conftest.py` に集約:
  - `pytest.fixture _StubAnalyzer`, `_ScriptedLLM`, `_ScriptedLLMClient`
  - `ingest_config_factory(tmp_path, **overrides) -> IngestConfig`
  - `tmp_samples_dir(tmp_path, content_map: dict[str, bytes | str])`
- `test_resume_after_crash.py`:
  - 既存成功 run の後、もう一度 ingest → wiki cache hit
  - `.staging/` だけ残っている状態 → 次 run は rmtree で回収
  - `.backup/` だけ残っている状態 → 次 run で warning、継続 (もしくは fail fast + guidance、実装時決定)
  - `.failed/` 残 → doctor / dry-run が検出
- `test_large_corpus.py` (`@pytest.mark.skipif(not os.getenv("RUN_LARGE"))`):
  - 100 file / 合計 500 KB を generator で作り、ingest が 60秒以内に完走
  - entity count の再現性 (2回 run で同数)
- 既存 `test_ingest.py:260-280`, `test_smoke.py:196-236` は U6 が無修正で通過するよう配慮
- Cross-device rename テストは実 mount 不要、`monkeypatch.setattr(os, "replace", raise_exdev_once)` で simulation

**Patterns to follow:**
- `tests/test_smoke.py:48-77` の `_ingest_fixture`
- `RUN_LLM=1` / `RUN_OLLAMA=1` env gating (`tests/test_smoke.py:281,299`)
- Fixture bytes 書き出し: `Path.write_bytes(b"...")`

**Test scenarios:**

_test_resume_after_crash.py:_
- Happy path: 2回目 ingest で wiki cache hit (既存 smoke テスト `test_wiki_second_run_skips_cache_hits` と重複しない観点: manifest path の resume)
- Error path: `.staging/` 残 + 新 ingest → finally-rmtree で回収、成功
- Error path: `.backup/` 残 + 新 ingest → warning ログ or fail fast + recovery 手順表示
- Error path: `.failed/` 残 → doctor が検出してレポート、次 ingest 実行は阻害しない

_test_large_corpus.py:_
- Happy path: 100 files × 5 KB → ingest 60秒以内
- Happy path: 同一 corpus 2回 ingest → entities_generated 一致 (決定論)
- Edge case: 1 file × 500 KB → sudachi byte-limit split path 通過

_Encoding variants (test_encoding.py 追記):_
- Shift-JIS + `--encoding auto` + charset-normalizer 導入 → 成功
- CP932 半角カナ → 成功
- UTF-8 BOM → `﻿` 除去済みで chunks.jsonl
- 混在 dir + `--encoding auto` → 全て成功
- 混在 dir + `--encoding utf-8` → Shift-JIS ファイルは `skipped_files.jsonl` に

_Atomic swap (test_atomic_swap_hardening.py, U6 で作成):_
- EXDEV simulation
- EACCES simulation
- `--verify-swap` post-swap mutation → mismatch

**Verification:**
- `pytest tests/` 全 pass (RUN_LARGE / RUN_LLM gate 除く)
- `RUN_LARGE=1 pytest tests/test_large_corpus.py` が ローカルで pass
- `grep -c "_StubAnalyzer" tests/*.py | grep -v conftest.py | grep -v ":0$"` が空 (conftest 以外に定義無し)

---

- [ ] U9. **RAG 検索評価ベンチマークスクリプト (chunking params regression 用)**

**Goal:** chunking params (chunk_size / overlap) × wiki on/off の組み合わせを、recall@5 / recall@10 / MRR@10 / nDCG@10 で比較できるスタンドアロンスクリプトを提供。**用途は自前 corpus 上でのチューニング / regression 検出** であり、ツール間比較ではない (README で positioning 明示)。

**Requirements:** R13

**Dependencies:** U7 (`[bench]` extra、`ranx` soft import)

**Files:**
- Create: `scripts/bench.py`
- Create: `samples/qrels.jsonl` (手書きで scaffolding、`samples/expected.yaml` の既存エントリを初期シードとして使う)
- Create: `tests/test_bench.py`
- Modify: `pyproject.toml` (`[bench] = ["ranx>=0.3"]`)
- Modify: `README.md` (新セクション「RAG 検索評価ベンチマーク (チューニング用途)」)

**Approach (scope-trimmed):**
- `scripts/bench.py`:
  - CLI: `python scripts/bench.py <corpus_dir> --configs NAME1=chunk:512,overlap:64,wiki:on --configs NAME2=... [--qrels samples/qrels.jsonl] [--out bench_out/] [--json bench_report.json]` (**config list の JSON schema はファイル化しない。CLI 引数として複数 `--configs` を受け、実装内で parse — `bench_configs.json` の新規 mini-schema は採用しない**)
  - 各 config ごとに ingest を呼び (library API: `IngestRunner(cfg).run()`)、`bench_out/<name>/` に出力
  - qrels を load (JSONL: `{"qid": "q1", "query": "...", "relevant": [{"chunk_id": "c:123", "grade": 2}, ...]}`)
  - 各 query に対し `run_query_impl(out_dir, query, top_k=10)` を実行
  - run dict 構築: `{qid: {chunk_id: score}}`
  - `ranx.Qrels.from_dict(...)`, `ranx.Run.from_dict(...)`, `ranx.evaluate(qrels, run, ["recall@5", "recall@10", "mrr@10", "ndcg@10"])`
  - stdout に table (rich が import できれば rich、なければ plain str)、`--json` 指定時 `bench_report.json` を emit
  - **paired t-test / `ranx.compare()` / significance marker は採用しない** (R13 範囲外の IR-researcher tooling。必要性が出たら別 Unit で対応)
- **`scripts/qrels_from_expected.py` 自動変換ヘルパーは作らない**。初回の `samples/qrels.jsonl` は `samples/expected.yaml` の既存エントリを見ながら手で起草 (既存 6-10 エントリ程度)。変換ヘルパー化は consumer が増えたら別 Unit
- `ranx` 未導入 → `ConfigError` (U1) に install hint `"pip install -e '.[bench]'"`
- 2件以上の config が必要 (単一 config はベンチマークにならない) → `ConfigError`
- README に positioning 明示: 「本ベンチマークは自前 corpus のチューニング / regression 検出用途。ツール間比較は対象外。外部データセット (JQaRA 等) との比較は別 release で検討」

**Patterns to follow:**
- `scripts/fast-ingest.sh` は shell wrapper、`bench.py` は Python モジュール
- `query.py:51-159` の `run_query_impl` をそのまま使う (subprocess 不要)
- JSONL emission

**Test scenarios:**
- Happy path: 2 config (c256_wiki_off, c512_wiki_on) × 3 query の tiny fixture で bench.py 実行 → stdout table 4 metrics × 2 rows
- Happy path: `--json` で `bench_report.json` emit、per-config metric dict (significance column は無し — scope-trimmed)
- Edge case: config 1 件のみ → `ConfigError`
- Edge case: qrels に chunk_id 不在 (壊れた qrels) → row ごとに `relevant=0` 扱い、crash 無し
- Error path: `ranx` 未導入 → exit 2 (`ConfigError`) with install hint `"pip install -e '.[bench]'"`
- Error path: 空 qrels → `ConfigError`
- Error path: corpus が 0 file → `ConfigError` (ingest 段階の no_input_files)
- Integration: `samples/qrels.jsonl` (手書き初期 scaffold) を bench.py に食わせて完走 — `scripts/qrels_from_expected.py` 自動変換ヘルパーは作らない (scope-trimmed)
- Integration: `tests/test_bench.py` が `[bench]` extra 入っていない env で skip

**Verification:**
- `pip install -e '.[bench]'` → `python -c 'import ranx'` 成功
- `python scripts/bench.py samples/ --configs c256=chunk:256,wiki:off --configs c512=chunk:512,wiki:on --qrels samples/qrels.jsonl --json bench_report.json` が成功
- README に bench 使用例 + 「チューニング / regression 用途、ツール間比較は対象外」の positioning 明示

---

## System-Wide Impact

- **Interaction graph**: 新しい `LorebookError` 階層は `ingest.py` / `analyzer.py` / `chunker.py` / `tfidf.py` / `ner.py` / `wiki.py` / `_swap.py` / `encoding.py` / `doctor.py` の全てが raise 側に立つ。CLI boundary (`cli.py` / `run_ingest`) で1箇所に catch。**`doctor` サブコマンドのみ exit code 空間 (0/1/18) を別管理** し ingest 系 (2-17) と衝突しない。`run_report.json` writer は `WikiStats` / `AggregationStats` / `ChunkerWarning` / `PhaseTimings` / `SkipReport` を集約 — 既存クラスには変更を加えず dataclass 拡張のみ
- **Error propagation**: domain レベルで raise → IngestRunner の phase try/except でラップ (必要に応じて原因 exception を `__cause__` として連結) → CLI boundary で `LorebookError.exit_code` 参照 → `IngestResult.errors` に `{class, message, context}` 記録 → `run_report.json.exit_reason` に書き出し
- **State lifecycle risks**: sibling dir 増殖: `<output>.staging/`, `<output>.backup/`, `<output>.failed/`, `<output>.swap-tmp/` (EXDEV 時のみ), `<output>/skipped_files.jsonl` (条件付き), `<output>/run_report.json` (常時)。crash 時の残骸ハンドリングは `doctor` で検出可能にする
- **API surface parity**:
  - Exit codes: 既存 (2/3/4/5/6/10) 後方互換、新規 ingest 系 (11-17) + doctor (0/1/18) 追加。`cli.py` + `ingest.py` の重複文字列を単一ソースに統合。**exit 3 = `LLMBackendUnavailableError` (preflight)、exit 6 = `WikiGenerationError` (runtime) を明確分離**
  - CLI flags: 新規 `--recursive`, `--glob`, `--encoding`, `--verify-swap`, `--dry-run`, `doctor` サブコマンド。既存 flag は無変更
  - Output files: `chunks.jsonl` 無変更、`entities/*.md` 無変更、`analyzer.json` 無変更、`log.md` 無変更、**`ingest_result.json` 削除 (breaking)**、**`run_report.json` 新規**、`skipped_files.jsonl` 新規 (条件付き、skip 0 件時は作成しない)、`.swap.manifest.sha256` 新規 (`--verify-swap` 時のみ staging 内に生成 → スワップ後 target に移動 → 検証成功後に削除して output dir を clean に保つ)
  - Python API: `IngestConfig` に新フィールド追加 (default 値で後方互換)、`IngestResult` に `skips`/`phase_timings`/`entities_generated` 追加 (既存 field は維持、`skipped_files` は `@property` 化)
- **Integration coverage**: cross-device rename、resume/crash、混在エンコーディング、大容量 (100-500 files)、`--verify-swap` post-swap mutation、`doctor` env probe、bench tiny fixture
- **Unchanged invariants**:
  - `chunks.jsonl` スキーマ (下流 RAG 消費者契約、prior plan の R2-R8)
  - `analyzer.json` スキーマ (query / lint が再構築に使用)
  - `entities/*.md` frontmatter (下流消費者契約)
  - `log.md` 形式 (human-readable、並存)
  - 既存 exit code 値 (2/3/4/5/6/10)
  - 既存 CLI 引数の位置と名前
  - 既存 LLM client 契約 (`anthropic_client.py` / `ollama_client.py` / `openai_client.py`)

---

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `ingest_result.json` 削除が CI パイプラインや外部ツールを破壊 | Low (v0.1.0、外部消費者ほぼ不在) | High | README に Breaking Change 明記、commit message に移行ガイド、`schema_version: 1` で将来の互換性担保。public release 直後のタイミング故、product-lens レビューで 1 リリース alias 維持の再検討余地あり (Open Questions → Product-lens 課題参照) |
| charset-normalizer の CJK 短文検出精度低下 (issue #121) | Medium | Medium | `utf-8-sig` strict 先行、短ファイル (<100 bytes) は detector 呼ばず明示要求、chaos 閾値 0.3 で保守的運用。本 CLI は novel/aozora 等 legacy 日本語 corpus 入力が現実的な分布 (web 99% UTF-8 の base rate に依存しない) のため `[full]` extra を **推奨 install** として README で前面に出す |
| warning 文字列 `"utf-8 decode failed"` が `[full]` extra の有無でテスト結果差 | Medium | Medium | U3 の既存 fixture test を `--encoding utf-8` 明示に書き換えて [full] 有無に非依存化 (U3 Test scenarios → Regression 項目) |
| macOS iCloud Drive / 外部マウント上の `os.replace` が EXDEV を出す (非 POSIX FS) | Medium | Medium | EXDEV fallback を既存テストで検証、README に「ローカル SSD 上の path 推奨」を明記 |
| parent-dir fsync が tmpfs / NFS で失敗 | Medium | Low | try/except で warning 化、swap 本体は継続 (durability は元々 best-effort と明記) |
| 新 exit code 11-17 + doctor 18 が既存 subprocess-integration スクリプトで想定外 | Low | Medium | 追加 (addition-only) であり既存 0/2/3/4/5/6/10 を保持、README に全テーブル。exit 3/6 の preflight/runtime 明確分離で既存挙動と一致 |
| `ProgressReporter` の phase boundary instrumentation 漏れで失敗 phase が 0.0s のまま run_report に出る | Medium | Medium | U5 Approach で 6 phase を try/finally wrap で囲むことを scope 内明記。失敗時も duration を記録 |
| `--verify-swap` の期待値乖離 (同一 FS 上では検出価値限定的) | Medium | Low | U6 Approach に意味論明記、README に「同一 FS 上は tautological、主な検出対象は EXDEV + I/O bit-flip」と docstring レベルで expectation setting |
| `--verify-swap` の SHA-256 計算が大容量コーパスで遅い | Medium | Low | opt-in で明示的、`run_report.phase_durations_seconds.swap` で可観測 |
| ranx の Numba JIT コンパイルが初回遅い | Low | Low | `[bench]` extra で opt-in、first-run 遅延は bench スクリプトで許容 |
| `_StubAnalyzer` / `_ScriptedLLM` の conftest 移動が既存テストの import を壊す | Medium | Low | U2 で前倒し実施して U3-U6 テストが最初から consolidated fixture を前提に書ける |
| `doctor` サブコマンドで外部 network probe (Ollama) が slow path | Low | Low | `--skip-network` フラグで opt-out、デフォルトは short timeout (5秒)。flag の実装是非は Deferred to Implementation |

---

## Success Metrics

- R1-R13 が全て test-covered (U1-U9 の Test scenarios で enumerate)
- `lorebook-chunker ingest samples/ out/` 成功時: `out/run_report.json` 存在、schema_version=1、必須キー全部
- `lorebook-chunker ingest samples/ out/` 失敗時: `out/run_report.json` 存在、`exit_reason.class` が `LorebookError` 派生クラス名
- `lorebook-chunker doctor` clean env で 2 秒以内に exit 0
- `lorebook-chunker ingest --dry-run samples/ out/` で `out/` が作成されず、stdout JSON に `files_discovered` / `first_file_encoding_probe`
- 既存 `pytest tests/` が全 pass (regression なし)
- `README.md` に: exit code テーブル、atomic swap contract、5 プロファイル、bench 使用例
- `python scripts/bench.py samples/ --configs <config>.json` が完走し recall@10 / MRR@10 / nDCG@10 を出力

---

## Phased Delivery

Phase 境界は緩い (hard gate ではなく land order の推奨)。各 Phase 内は並行可、Phase 跨ぎは前 Phase の foundation に依存。

### Phase 1 — Foundation
- **U1** Error taxonomy (他全て前提)

### Phase 2 — Input pipeline
- **U2** Input discovery options
- **U3** Encoding detection
- **U4** SkipReport + `skipped_files.jsonl`

### Phase 3 — Output contract
- **U5** Phase timings + `run_report.json` (U4 の SkipReport を参照)
- **U6** Atomic swap hardening

### Phase 4 — Operator-facing
- **U7** `--dry-run` / `doctor` + 依存プロファイル docs
- **U8** 統合テスト (U2-U6 の挙動を verify)
- **U9** RAG eval benchmark (独立性が高いので U7 と並行可)

---

## Documentation Plan

- **`README.md`** 新セクション/更新:
  - インストール (U7): 5 プロファイル table
  - Exit codes (U1): `errors.describe_exit_codes()` に対応した全コード表
  - Atomic swap contract (U6): same-fs / cross-device / durability スコープ、sibling dir の意味
  - 実行レポート (U5): `run_report.json` スキーマ例、主要フィールド説明
  - RAG 評価ベンチマーク (U9): 使い方、qrels 形式、出力例
- **CHANGELOG**: repo に CHANGELOG.md は未存在。本リリース用 commit message に Breaking Change として `ingest_result.json 削除` と `run_report.json 追加` を明記。必要なら別 Issue で CHANGELOG.md 導入を提起
- **CLI `--help` epilog**: `U1` で単一ソース化、全サブコマンドが最新 exit code を epilog で表示
- **`docs/plans/`**: 本計画 (live)。実装完了時に `status: completed` に更新
- **`docs/solutions/` (新規候補)**: リリース後、以下2件を solution 化検討: (a) 2026 年 Python CLI のエラー taxonomy パターン、(b) atomic directory swap contract の書き方

---

## Sources & References

- **Origin Issue**: [wfukatsu/lorebook-chunker#1 — Improve production readiness](https://github.com/wfukatsu/lorebook-chunker/issues/1)
- **Prior plan**: `docs/plans/2026-04-21-001-feat-japanese-rag-chunking-plan.md` (R1-R26 + R8b + R18b、Unit 1-12)
- **Prior requirements**: `docs/brainstorms/2026-04-21-japanese-rag-chunking-requirements.md`
- Related code anchors (全 repo-relative):
  - `src/lorebook_chunker/cli.py:8-11, 13-37, 43-115` — identity banner, epilogs, argparse
  - `src/lorebook_chunker/ingest.py:257-265, 288-293, 304-558, 564-567, 602-624, 660-702, 720-729, 794-823` — IngestResult / top-level try/except / discovery / swap / log writer / result JSON
  - `src/lorebook_chunker/schema.py:119-128` — 既存 custom exceptions
  - `src/lorebook_chunker/llm/__init__.py:28-41` — LLM error 階層
  - `src/lorebook_chunker/wiki.py:157-170, 223-260` — WikiStats + atomic write reference
  - `src/lorebook_chunker/analyzer.py:629-646` — atomic write reference
  - `src/lorebook_chunker/ner.py:37-43` — AggregationStats
  - `src/lorebook_chunker/chunker.py:29-35` — ChunkerWarning
  - `src/lorebook_chunker/progress.py:1-138` — ProgressReporter (phase timing source)
  - `src/lorebook_chunker/query.py:51-159` — `run_query_impl` (bench で再利用)
  - `tests/test_ingest.py:17-89, 209-233, 260-280` — stub patterns + existing skip/swap tests
  - `tests/test_smoke.py:48-77, 112-160, 196-236, 253-275, 281, 299` — `_ingest_fixture`, query tests, crash test, env gates
  - `scripts/fast-ingest.sh:70-73, 102-104` — 既存 env validation pattern
  - `pyproject.toml:17-45` — 依存とプロファイル
  - `README.md:241-307, 480, 529` — install / swap / single-writer 前提
- External references (Key Technical Decisions 節に詳細):
  - [clig.dev CLI Guidelines](https://clig.dev/)
  - [pip status_codes.py](https://github.com/pypa/pip/blob/main/src/pip/_internal/cli/status_codes.py)
  - [charset_normalizer (jawah)](https://github.com/jawah/charset_normalizer) / [issue #121 CJK 精度](https://github.com/jawah/charset_normalizer/issues/121)
  - [os.replace 公式 doc](https://docs.python.org/3/library/os.html#os.replace) / [alexwlchan Atomic cross-filesystem moves](https://alexwlchan.net/2019/atomic-cross-filesystem-moves-in-python/) / [Calvin Loncaric durable writes](https://calvin.loncaric.us/articles/CreateFile.html)
  - [ranx](https://github.com/AmenRa/ranx) / [JQaRA (将来統合候補)](https://github.com/hotchpotch/JQaRA)
