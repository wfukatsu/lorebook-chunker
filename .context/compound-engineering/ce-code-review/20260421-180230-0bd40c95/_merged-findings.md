# Merged Findings — Run 20260421-180230-0bd40c95

13 reviewers dispatched (11 structured, 2 narrative). After confidence gate (≥0.60; P0 at 0.50+) and dedup, the merged set below preserves the highest severity and confidence across reviewers, with cross-reviewer agreement boosted by +0.10 (capped at 1.00).

No finding is pre-existing — every file in scope is new code.

## P0 — Critical (1)

**F-001 · `row_index` in chunks.jsonl is per-document, not global** — breaks RAG retrieval contract.
- Files: `src/chunking/chunker.py:82-91`, `src/chunking/ingest.py:174-191`
- Reviewers: api-contract (0.97), testing (gap, 0.95) → merged conf 1.00
- Route: safe_auto → review-fixer
- Verified: `chunk_document` uses `enumerate(final_chunks)` starting at 0 per file; `ingest.py` concatenates into `all_chunks` and feeds the whole list to TF-IDF. Every chunk from the 2nd document onward has a `row_index` that points to the wrong matrix row.
- Fix: enumerate across the global `all_chunks` list after accumulation, not inside `chunk_document`. Alternatively set `row_index=None` inside the chunker and assign globally in ingest.

## P1 — High (21 merged findings)

### Correctness / data integrity
- **F-002 · allow_pickle=True in TfidfBuilder.load → RCE** — `src/chunking/tfidf.py:129`. Reviewer: security (0.95, with live PoC). Fix: store vocabulary as `str_` dtype + config as utf-8 bytes; load with `allow_pickle=False`.
- **F-003 · Pre-flight bool counted in `max_llm_calls` budget** — `src/chunking/wiki.py:385-386`. `budget_consumed = stats.preflight_called + stats.attempted` is `True + 0 = 1`; `--max-llm-calls=1` exhausts budget before entity 0; `--max-llm-calls=0` still makes one real LLM call. Reviewers: correctness (0.92), adversarial ×2 (0.92, 0.88), kieran-python (0.63) → merged 1.00. Fix: `budget_consumed = stats.attempted` (and count preflight separately for accounting).
- **F-004 · `rebuild_vectorizer` `.idf_` attribute injection likely broken on sklearn ≥1.3** — `src/chunking/tfidf.py:158-179`. `TfidfVectorizer.idf_` is a property; attribute assignment shadows silently while internal `_tfidf` still uses the dummy-fit IDF. Every query score is wrong. Reviewers: adversarial (0.78), kieran-python (0.80) → merged 0.88. gated_auto → downstream-resolver, requires_verification.
- **F-005 · Staging directory leaked on LLMPermanentError** — `src/chunking/ingest.py:252-255`. `except LLMPermanentError` returns before the outer `except Exception` cleanup. Reviewer: correctness (0.95). safe_auto → review-fixer.
- **F-006 · entities/*.md frontmatter missing required fields** — `src/chunking/wiki.py:253-264`. Plan R15 requires `status`, `last_attempt_at`, `chunk_ids`; the implementation omits all three. Reviewer: api-contract (0.92). safe_auto → review-fixer.
- **F-007 · entities/manifest.json `failure_reason` not in plan schema** — `src/chunking/wiki.py:110-120, 216`. `ManifestEntry(**raw)` in `load()` will also TypeError on any future field addition. Reviewer: api-contract (0.88). gated_auto → downstream-resolver (contract decision).
- **F-008 · lint exit code is 0/1, not the specified 0/1/2** — `src/chunking/lint.py:293`. Warning-only runs silently exit 0. Reviewers: api-contract (0.85), cli-readiness (0.95) → merged 1.00. safe_auto → review-fixer.
- **F-009 · `--skip-wiki` still writes `index.md`** — `src/chunking/ingest.py:258`. Line is outside the `if not self.cfg.skip_wiki:` guard. Reviewer: project-standards (0.95). safe_auto → review-fixer.
- **F-010 · `samples/01_news.txt … 06_fiction.txt` required by Unit 11 are absent** — `samples/`. Only `README.md` + `expected.yaml` present. Reviewer: project-standards (0.95). manual → downstream-resolver.
- **F-011 · `_atomic_swap` cross-filesystem + crash windows can destroy prior output** — `src/chunking/ingest.py:317-328`. Three failure shapes: (a) `copytree` fallback fails → no backup restore; (b) crash between `target.rename(backup)` and `staging.rename(target)` + next run's unconditional `rmtree(backup)` → permanent data loss; (c) Ctrl-C on staging → next run's unconditional `rmtree(staging)` throws away checkpointed wiki pages. Reviewers: reliability (0.90), adversarial ×2 (0.88, 0.87), correctness (0.62) → merged 1.00. gated_auto → downstream-resolver, requires_verification (atomic-swap strategy is a contract decision).
- **F-012 · Entity names `A/B` vs `A\B` collide after sanitization** — `src/chunking/ner.py:140`. `sanitize_entity_filename` collapses `/` and `\` to `_` with no uniqueness suffix, so `_record_success` silently overwrites the earlier entity's wiki page. Reviewer: adversarial (0.83). safe_auto → review-fixer.
- **F-013 · LLM response containing bare `---` line splits wiki YAML frontmatter** — `src/chunking/wiki.py` (page renderer). Downstream YAML parsers will treat the injected `---` as frontmatter end. Reviewer: adversarial (0.85). safe_auto → review-fixer (fence the body or escape `^---$`).
- **F-014 · Concurrent ingest on cross-filesystem mount silently overwrites outputs** — `src/chunking/ingest.py`. Two processes share staging/backup naming; outcomes are non-deterministic and neither errors. Reviewer: adversarial (0.82). manual → downstream-resolver (doc + sentinel file).
- **F-015 · Systemic-failure abort writes manifest to staging, then IngestRunner rmtree's staging** — `src/chunking/ingest.py:278-283` + `src/chunking/wiki.py`. Exit_code=6 path loses all wiki work including the manifest the abort tried to preserve. Reviewer: adversarial (0.76). safe_auto → review-fixer (move manifest into backup-on-failure path).
- **F-016 · Ollama client has no timeout — hangs indefinitely on stuck daemon** — `src/chunking/llm/ollama_client.py:54-67`. Reviewer: reliability (0.85). safe_auto → review-fixer.

### Agent-readiness / contract
- **F-017 · `query` output is human prose only — no `--format json`** — `src/chunking/query.py:151-157`. Agents capturing stdout must regex-parse. Reviewers: cli-readiness (0.97), agent-native. manual → downstream-resolver.
- **F-018 · `test_paraphrase_query_recorded` is a no-op test** — `tests/test_smoke.py:126-140`. Asserts file existence, never content. Reviewer: testing (0.82). safe_auto → review-fixer (replace with a real assertion).

### Type safety / silent correctness
- **F-019 · `ChunkRecord.top_keywords/entities` typed `list[dict[str, Any]]`; `AnalyzerConfig.strict_match/compat_match/tfidf` all `dict[str, Any]`** — `src/chunking/schema.py:31-32, 56-59`. These **are** the public contract. Reviewer: kieran-python (0.92, 0.90). safe_auto → review-fixer (introduce `KeywordEntry` TypedDict, reuse existing `EntityMention`, add `StrictMatchDict` + `CompatMatchDict`).
- **F-020 · `EntityStatus = str` should be `Literal["success","failed","budget_skipped"]`** — `src/chunking/wiki.py:59`. Cache-skip branching depends entirely on these string literals. Reviewers: kieran-python (0.88), maintainability (0.87) → merged 0.98. safe_auto → review-fixer.
- **F-021 · `_ginza_model_version` bare `except` silently returns `"unknown"` — corrupts source_hash cache keys** — `src/chunking/ingest.py:296-303`. A real failure would silently cause every entity to re-generate (or never-invalidate) on Ginza upgrades. Reviewers: kieran-python (0.82), maintainability (0.72) → merged 0.92. gated_auto → downstream-resolver.
- **F-022 · Broad `except Exception` in `IngestRunner.run` turns every programming error into `exit_code=10`** — `src/chunking/ingest.py:278-284`. Reviewer: kieran-python (0.85). gated_auto → downstream-resolver (narrowing may unmask legitimate bugs that need separate handling).

## P2 — Moderate (~30)

Grouped. Full detail in per-reviewer artifact files.

### Supply chain / packaging
- **F-023 · `ja-ginza-electra` wheel URL has no `#sha256=`; numpy has no upper bound** — `pyproject.toml:17,22`. Reviewers: security (0.80), kieran-python (0.72) → merged 0.90. safe_auto → review-fixer.
- **F-024 · `sanitize_entity_filename` strips `/`, `\` but not `.`** — `src/chunking/ner.py:140`. All-dot names produce `..md` or dot-only stems. Reviewer: security (0.75). safe_auto → review-fixer.

### Correctness / contract (medium)
- **F-025 · `normalize_text` missing global `.strip()`** — `src/chunking/normalize.py:20`. Violates plan normalization contract and the docstring example; `test_mixed_real_world` currently asserts the buggy behavior. Reviewer: correctness (0.90). safe_auto → review-fixer, requires_verification (existing test needs updating).
- **F-026 · `vocab.npz` `config_json` loaded via `str(0-d numpy array)`** — `src/chunking/tfidf.py:141`. Undocumented internal; use `.item()`. Reviewer: correctness (0.78). safe_auto → review-fixer.
- **F-027 · `LLMPermanentError` mid-run demoted to per-entity failure; systemic abort never fires for <5 entity corpora** — `src/chunking/wiki.py:519-521`. Reviewer: reliability (0.82). gated_auto → downstream-resolver.
- **F-028 · `query._load_chunks` has no error handling — corrupt line crashes query** — `src/chunking/query.py:132-137`. Reviewer: reliability (0.88). safe_auto → review-fixer.
- **F-029 · analyzer.json tfidf block emits `sublinear_tf` and `top_keywords` not in plan schema** — `src/chunking/tfidf.py:149-155`. Reviewer: api-contract (0.75). gated_auto.
- **F-030 · identity banner only in `--help`, not emitted to stderr on ingest start (R9 intent)** — `src/chunking/ingest.py:424-458`. Reviewers: api-contract (0.72), cli-readiness. safe_auto → review-fixer.
- **F-031 · Deleted manifest but `entities/*.md` still present → regenerated pages overwrite human edits** — `src/chunking/wiki.py:150-156`. gated_auto → human (policy decision). Reviewer: adversarial (0.80).
- **F-032 · All-identical corpus → every term exceeds `max_df=0.95` → empty vocab; user sees misleading "vocab miss" message** — lint/tfidf boundary. Reviewer: adversarial (0.81). advisory → human.
- **F-033 · Zero-width joiner (U+200D) in entity names splits one logical entity into two manifest keys** — NFKC doesn't remove ZWJ. Reviewer: adversarial (0.73). advisory → human.
- **F-034 · `char_start/char_end` on chunks are post-overlap snap points, not original sentence starts** — chunker.py. Callers assuming "natural sentence boundary" will be off by overlap. Reviewer: adversarial (0.68). advisory → human (documentation).

### CLI readiness
- **F-035 · `lint` has no `--format json`** — `src/chunking/lint.py:296-305`. Reviewer: cli-readiness (0.95). manual.
- **F-036 · `ingest` success confirmation on stderr only, no per-run JSON summary** — `src/chunking/ingest.py:452-457`. Reviewer: cli-readiness (0.90). manual.
- **F-037 · ingest exit codes 2-10 undocumented in help + README** — `src/chunking/ingest.py:122-280`. Reviewer: cli-readiness (0.92). safe_auto → review-fixer (help epilog + README section).
- **F-038 · `query` zero-hit / OOV returns exit 0 — indistinguishable from legitimate no-match** — `src/chunking/query.py:148-150`. Reviewer: cli-readiness (0.88). manual.

### Performance
- **F-039 · `top_keywords_per_chunk` Python-loop with `getrow()` O(N)** — `src/chunking/tfidf.py:79-94`. Reviewer: performance (0.85). safe_auto → review-fixer (operate on `data`/`indices`/`indptr`).
- **F-040 · `rebuild_vectorizer` runs full Ginza tokenization over joined vocab on every query** — `src/chunking/tfidf.py:175-177`. Reviewer: performance (0.82). gated_auto (fix overlaps with F-004).
- **F-041 · `aggregate_entities` O(E·C)** — `src/chunking/ner.py:116-118`. Reviewer: performance (0.80). safe_auto.
- **F-042 · `by_id` dict rebuilt per entity in two WikiGenerator methods** — `src/chunking/wiki.py:294-313, 529-533`. Reviewers: maintainability (0.80), performance (0.78) → 0.90. safe_auto → review-fixer.
- **F-043 · Pipeline runs Ginza 3× per chunk (iter_sentences, tokenize_for_tfidf, iter_entities)** — `src/chunking/ingest.py:167-213`. Reviewer: performance (0.75). manual → downstream-resolver (combined analyze method).

### Maintainability
- **F-044 · Dead CLI flag `--config`** — `src/chunking/cli.py:32`. Reviewer: maintainability (0.97). safe_auto (delete).
- **F-045 · Duplicate `_load_chunks` in query.py and lint.py** — `src/chunking/query.py:132` + `src/chunking/lint.py:208`. Reviewer: maintainability (0.95). safe_auto.
- **F-046 · `QueryAnalyzer` Protocol unused** — `src/chunking/query.py:17-18`. Reviewer: maintainability (0.93). safe_auto (delete).
- **F-047 · `IngestAnalyzer` Protocol duplicates `SupportsIterEntities` without cross-ref** — `src/chunking/ingest.py:42-55`. Reviewer: maintainability (0.82). manual (design decision).

### Portability
- **F-048 · `_posix_relative` uses `PurePosixPath` on native OS paths — breaks on Windows** — `src/chunking/chunker.py:267`. Reviewer: project-standards (0.82). safe_auto (use `Path(...).relative_to(...).as_posix()`).

### Type discipline (P2 tier)
- F-049 `_append_log_md` `agg_stats` untyped (`ingest.py:374-375`, 0.80)
- F-050 Broad except in `_preflight` (`wiki.py:452-454`, 0.78)
- F-051 `aggregate_entities` dict accumulator + 2× `type: ignore` (`ner.py:91-112`, 0.76)
- F-052 `_NoopLLM.generate` missing return type (`ingest.py:417`, 0.75)
- F-053 Broad except in `_verify_ne_extension` (`analyzer.py:110`, 0.74)
- F-054 `__import__` inside fixture body (`tests/test_smoke.py:60-63`, 0.70)

## P3 — Low (9)

- F-055 `buf_len` underestimates true chunk length (inter-sentence gaps uncounted) — `chunker.py:158` (0.80)
- F-056 Stray BIO I-tags silently dropped — `analyzer.py:154` (0.70)
- F-057 `analyzer.save` writes without fsync — `analyzer.py:217-222` (0.72)
- F-058 `IngestRunner` early-exit paths 5/6 don't clean staging — `ingest.py:195-255` (0.65, complements F-005)
- F-059 `AnalyzerFactory` / `LLMClientFactory` aliases each used exactly once — `ingest.py:57-58` (0.75)
- F-060 `lint` tuning thresholds not exposed as CLI flags — `lint.py:51,61` (0.83)
- F-061 `_detect_degenerate_rows` Python-level sparse iteration — `lint.py:228-237` (0.65)
- F-062 `--top-k 0` empty slice indistinguishable from no-match — `query.py` (0.75)
- F-063 `tiny_ja_corpus` fixture defined but never consumed — `tests/conftest.py:7-14` (0.65)

## Coverage / testing gaps (not findings, for the report)

- No test covers atomic-swap cross-filesystem OSError fallback branch (F-011 relevant).
- No test covers `analyzer.json` version-mismatch on query path in Ginza-free environment.
- `budget_skipped` + `--retry-failed=True` interaction (distinct branch from `status=failed`) untested.
- LLM retry-exhaustion test does not verify `failure_reason` or absent `.md` file.
- Smoke tests with `--skip-wiki=True` never assert `manifest.json` schema.
- `test_manifest_atomic_write_on_failure` only exercises happy path; never injects a write failure.
- Unicode NFD voiced-kana / composed/decomposed variants untested.
- `row_index` not verified globally sequential across multi-file smoke corpus (directly related to F-001).
- CLI smoke test does not cover `--skip-wiki` / `--max-llm-calls` argument parsing.
- `ingest` exit codes 4, 5, 6 have no test coverage.
- `query.py` exit code 3 (row-count mismatch guard) untested.

## Residual risks

- `ManifestStore.load` uses `ManifestEntry(**raw)` — any future schema field addition raises `TypeError` on old manifests with no migration path.
- `OllamaLLMClient` classifies most exceptions as `LLMRetryableError`; permanent server errors are retried 3× unnecessarily (overlaps with F-016).
- `query.py` cosine similarity assumes L2-normalized rows; if `norm=None` were ever used at ingest, rankings would be silently wrong.
- `_StubAnalyzer.tokenize_for_tfidf` (tests) does not call `normalize_text`, so NFKC dedup behavior in the real pipeline is never exercised under the stub-based tests.
- Fixture corpus is very small (~10 lines per file); chunk-boundary semantics on realistic Japanese prose are not asserted.
- No regression test verifies `compute_source_hash` actually changes when `PROMPT_TEMPLATE_VERSION` is bumped.

## Protected-artifact rule

No reviewer flagged protected artifacts (`docs/brainstorms/*`, `docs/plans/*`, `docs/solutions/*`) for deletion or gitignore. Rule was respected.

## Plan inferred? Explicit.

Plan `docs/plans/2026-04-21-001-feat-japanese-rag-chunking-plan.md` was present and explicitly referenced in the commit series. Multiple P1/P0 findings are *plan contract violations* (F-001, F-003, F-006, F-007, F-008, F-009, F-010, F-030). When the plan is the contract of record, any divergence is at least P1 even if the code independently "works".

## Requirements completeness (explicit plan)

| Req | Unit | Status | Notes |
|-----|------|--------|-------|
| R1-R8 | 2-4 | partially met | F-029: analyzer.json emits schema fields not in plan; F-001: row_index bug |
| R8b | 8 | met | destructive regen via atomic swap is implemented (quality concerns in F-011) |
| R9 | 8 | partially met | F-030: identity banner not emitted; F-010: samples/*.txt missing |
| R10 | 9 | partially met | F-017, F-038: query CLI ships but not machine-readable; also F-004/F-040 query-path correctness |
| R11-R12 | 10 | partially met | F-008: exit codes wrong; F-035: no `--format json`; F-060: thresholds not exposed |
| R13-R14 | 5 | met | subject to F-012 (filename collision) and F-033 (ZWJ) |
| R15 | 7 | NOT met | F-006: wiki frontmatter missing `status`, `last_attempt_at`, `chunk_ids` |
| R16 | 8 | partially met | F-009: index.md written even with `--skip-wiki` |
| R17 | 8+7 | met | log.md writer present |
| R18, R18b | 7 | met | source_hash and retry logic present (but F-003 budget bug, F-021 silently unknown Ginza version, F-031 human-edit overwrite) |
| R19-R21 | 6 | met | LLM abstraction present (subject to F-016 Ollama timeout) |
| R22 | 7 | partially met | F-027: systemic-failure abort threshold broken for <5 entity corpora; F-003 budget bug |
| R23-R25 | 11 | NOT met | F-010: samples missing |
| R26 + Success Criteria | 12 | partially met | F-018: paraphrase test is no-op |
