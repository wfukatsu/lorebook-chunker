# Review Context — Run 20260421-180230-0bd40c95

## Mode
Interactive. Standalone full-audit review (no PR, no remote).

## Scope
- **Repo:** `/Users/wfukatsu/work/chunking`
- **Diff base:** `bf11cf8` (root commit; only `docs/brainstorms/…` and `docs/plans/…` exist at that commit)
- **Files in scope:** 42 files, ~5,530 insertions. Every source/test file is new. Reading the file in the working tree == reading the full diff for that file. Do **not** ask the user for a separate diff dump.
- **Working tree is clean.** What is on disk is what is under review.
- **Branch:** `main` (only branch, no remote).

## Intent
Ship the initial full implementation of a Japanese RAG preprocessing CLI per
`docs/plans/2026-04-21-001-feat-japanese-rag-chunking-plan.md`. All 12 planned
units are present:

1. project scaffolding + CLI entrypoint
2. analyzer module (normalize, schema, Ginza wrapper)
3. chunker (char-target + overlap + soft-split)
4. TF-IDF builder (unified `vocab.npz` persistence)
5. NER aggregation (threshold filtering + sanitization)
6. LLM client abstraction (Anthropic / Ollama / OpenAI placeholder)
7. wiki generator (cache, pre-flight, retry/budget guards)
8. ingest CLI (staging dir atomic swap orchestration)
9. query CLI (cosine top-k search)
10. lint CLI (fatal/warning/info tiers)
11. sample corpus + expected outputs
12. smoke tests (fixture corpus + skip-gated real-sample checks)

### Output contract (two tiers)
1. **RAG retriever inputs:** `chunks.jsonl` + `vocab.npz` + `analyzer.json`
2. **First-class entity knowledge base:** `entities/*.md` + `entities/manifest.json` + `index.md`

### Key technical decisions (from plan, relevant to review)
- `sparse_vec` is **not** inlined in `chunks.jsonl` — each chunk carries `row_index` that indexes into `vocab.npz`.
- `source_hash` in `entities/manifest.json` is the primary store for wiki retry decisions; individual wiki frontmatter is a redundant copy.
- Ginza NER goes through `token._.ne` (OntoNotes5 via BIO-tag merge) rather than `doc.ents`.
- `analyzer.json` distinguishes `strict_match.*` (must match byte-for-byte) from `compat_match.*` (major.minor).
- LLM call ordering is deterministic: `mention_count DESC → chunk_count DESC → entity_name ASC` so `--max-llm-calls` consumes high-value entities first.
- Normalization: NFKC + LF-only + trim + collapse; `char_start/char_end` in chunks refer to post-normalization offsets.

## Standards Files
No `CLAUDE.md` or `AGENTS.md` is present inside `/Users/wfukatsu/work/chunking`.
The parent workspace `/Users/wfukatsu/work/CLAUDE.md` exists but explicitly describes
`/Users/wfukatsu/work` as a collection of unrelated subprojects and does **not**
document conventions for the `chunking/` subdirectory. Project-standards review
should note the absence rather than inventing standards.

## Artifacts
- Full JSON output: `.context/compound-engineering/ce-code-review/20260421-180230-0bd40c95/<reviewer_name>.json`
- Return compact JSON (merge-tier only) to the orchestrator. Do **not** include
  `why_it_matters` or `evidence[]` in the return; those live in the artifact file.

## File List (42)
```
.gitignore
README.md
pyproject.toml
samples/README.md
samples/expected.yaml
src/chunking/__init__.py
src/chunking/__main__.py
src/chunking/analyzer.py
src/chunking/chunker.py
src/chunking/cli.py
src/chunking/ingest.py
src/chunking/lint.py
src/chunking/llm/__init__.py
src/chunking/llm/anthropic_client.py
src/chunking/llm/ollama_client.py
src/chunking/llm/openai_client.py
src/chunking/ner.py
src/chunking/normalize.py
src/chunking/query.py
src/chunking/resources/__init__.py
src/chunking/resources/preflight_prompt.txt
src/chunking/schema.py
src/chunking/tfidf.py
src/chunking/wiki.py
tests/__init__.py
tests/conftest.py
tests/fixtures/samples/01_news.txt
tests/fixtures/samples/02_tech.txt
tests/fixtures/samples/03_interview.txt
tests/fixtures/samples/04_fiction.txt
tests/test_analyzer.py
tests/test_chunker.py
tests/test_cli_smoke.py
tests/test_ingest.py
tests/test_lint.py
tests/test_llm_clients.py
tests/test_ner.py
tests/test_normalize.py
tests/test_query.py
tests/test_smoke.py
tests/test_tfidf.py
tests/test_wiki.py
```

## Reviewer Team (13)
Always-on: correctness, testing, maintainability, project-standards, ce-agent-native, ce-learnings-researcher
Cross-cutting: security (LLM API keys + arbitrary FS paths), performance (TF-IDF / NER throughput), api-contract (2 stable output contracts), reliability (retry / budget / atomic swap), adversarial (>>50 non-generated lines; external APIs; data mutations on disk), cli-readiness (3-subcommand CLI)
Stack-specific: kieran-python
