"""U9: `scripts/bench.py` の単体テスト.

`ranx` が未導入の環境では file 全体を skip する (bench extra は opt-in のため).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ranx が未導入なら file 全体を skip.
# exc_type=ImportError は pytest 9.1 の新デフォルト互換 (未指定時の警告抑制).
ranx = pytest.importorskip("ranx", exc_type=ImportError)

# bench.py を module として import するために scripts/ を path に追加する.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import bench  # type: ignore  # noqa: E402


# ---- Fixtures / helpers ----------------------------------------------------


_TINY_NEWS = (
    "スカラー商事は本日、アルファ合成との合併について正式に発表した。\n"
    "スカラー商事の田中社長は記者会見で、合併の背景について説明した。\n"
    "田中はアルファ合成との合併により、東京と大阪の事業基盤が強化されると述べた。\n"
    "アルファ合成の佐藤社長も合併に同意しており、経営統合の経緯は順調に進んでいる。\n"
    "両社は半年後に合併を完了する予定である。\n"
)

_TINY_TECH = (
    "スカラー商事の技術資料: ベータプロジェクトの概要を記述する。\n"
    "ベータプロジェクトは分散データベースを構築する取り組みである。\n"
    "プロジェクトの責任者は田中で、東京本社で開発が進む。\n"
)


@pytest.fixture
def tiny_corpus(tmp_path: Path, tmp_samples_dir) -> Path:
    return tmp_samples_dir(
        tmp_path / "corpus",
        {"01_news.txt": _TINY_NEWS, "02_tech.txt": _TINY_TECH},
    )


def _run_ingest_once_for_chunk_ids(corpus: Path, tmp_path: Path) -> dict[str, str]:
    """テスト用の reference ingest を回し、source -> chunk_id の map を返す.

    bench tests は chunk_ids を直接知らないため、qrels を書く前に 1 度 ingest
    を走らせて chunk_id を回収する.
    """
    from lorebook_chunker.ingest import IngestConfig, IngestRunner

    ref_out = tmp_path / "ref_out"
    cfg = IngestConfig(
        input_dir=corpus,
        output_dir=ref_out,
        skip_wiki=True,
        target_chars=256,
        overlap_chars=32,
        min_mentions=1,
        min_chunks=1,
        show_progress=False,
    )
    result = IngestRunner(cfg).run()
    assert result.exit_code == 0, f"ref ingest failed: {result.errors}"
    m: dict[str, str] = {}
    with (ref_out / "chunks.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            # 最初に出た chunk を source の代表とする
            m.setdefault(d["source"], d["chunk_id"])
    return m


def _write_qrels(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False))
            f.write("\n")


# ---- parse_config_spec ----------------------------------------------------


class TestParseConfigSpec:
    def test_full_spec(self) -> None:
        bc = bench.parse_config_spec("c512=chunk:512,overlap:64,wiki:on")
        assert bc.name == "c512"
        assert bc.chunk_size == 512
        assert bc.overlap == 64
        assert bc.skip_wiki is False

    def test_wiki_off(self) -> None:
        bc = bench.parse_config_spec("c256=chunk:256,overlap:32,wiki:off")
        assert bc.skip_wiki is True

    def test_defaults_when_keys_missing(self) -> None:
        bc = bench.parse_config_spec("x=chunk:200")
        assert bc.chunk_size == 200
        assert bc.overlap == 100  # default
        assert bc.skip_wiki is True  # default (wiki:off)

    def test_unknown_key_raises(self) -> None:
        from lorebook_chunker.errors import ConfigError

        with pytest.raises(ConfigError) as exc:
            bench.parse_config_spec("x=foo:bar")
        assert exc.value.context.get("reason") == "unknown_config_key"

    def test_bad_wiki_value_raises(self) -> None:
        from lorebook_chunker.errors import ConfigError

        with pytest.raises(ConfigError):
            bench.parse_config_spec("x=wiki:maybe")

    def test_missing_name_raises(self) -> None:
        from lorebook_chunker.errors import ConfigError

        with pytest.raises(ConfigError):
            bench.parse_config_spec("=chunk:200")


# ---- load_qrels ------------------------------------------------------------


class TestLoadQrels:
    def test_happy_path(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        _write_qrels(
            path,
            [
                {"qid": "q1", "query": "test", "relevant": [{"chunk_id": "a", "grade": 2}]},
            ],
        )
        entries = bench.load_qrels(path)
        assert len(entries) == 1
        assert entries[0].qid == "q1"
        assert entries[0].relevant == {"a": 2}

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        from lorebook_chunker.errors import ConfigError

        path = tmp_path / "q.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError) as exc:
            bench.load_qrels(path)
        assert exc.value.context.get("reason") == "empty_qrels"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from lorebook_chunker.errors import ConfigError

        with pytest.raises(ConfigError) as exc:
            bench.load_qrels(tmp_path / "nonexistent.jsonl")
        assert exc.value.context.get("reason") == "qrels_not_found"

    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        from lorebook_chunker.errors import ConfigError

        path = tmp_path / "q.jsonl"
        path.write_text("not-json\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            bench.load_qrels(path)


# ---- evaluate_run ----------------------------------------------------------


class TestEvaluateRun:
    def test_perfect_match(self) -> None:
        qrels = [bench.QrelEntry(qid="q1", query="test", relevant={"a": 2})]
        run = {"q1": {"a": 0.9, "b": 0.1}}
        metrics = bench.evaluate_run(qrels, run)
        assert set(metrics.keys()) == set(bench.METRICS)
        assert metrics["recall@5"] == 1.0
        assert metrics["mrr@10"] == 1.0

    def test_no_match_chunk_id_not_in_run(self) -> None:
        # 壊れた qrels: chunk_id が run に存在しない → metric は 0 に degrade.
        qrels = [bench.QrelEntry(qid="q1", query="test", relevant={"ghost": 2})]
        run = {"q1": {"a": 0.9, "b": 0.5}}
        metrics = bench.evaluate_run(qrels, run)
        assert metrics["recall@5"] == 0.0
        assert metrics["ndcg@10"] == 0.0

    def test_empty_run_for_query(self) -> None:
        qrels = [bench.QrelEntry(qid="q1", query="test", relevant={"a": 2})]
        run = {"q1": {}}
        metrics = bench.evaluate_run(qrels, run)
        # ranx は empty run を 0 metric で扱う
        assert metrics["recall@5"] == 0.0


# ---- bench.run end-to-end --------------------------------------------------


class TestBenchRun:
    def test_happy_path_emits_table_and_json(
        self, tiny_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ref = _run_ingest_once_for_chunk_ids(tiny_corpus, tmp_path)
        qrels_path = tmp_path / "q.jsonl"
        _write_qrels(
            qrels_path,
            [
                {
                    "qid": "q1",
                    "query": "合併の背景 スカラー商事 田中",
                    "relevant": [{"chunk_id": ref["01_news.txt"], "grade": 2}],
                },
                {
                    "qid": "q2",
                    "query": "ベータプロジェクト 田中 東京本社",
                    "relevant": [{"chunk_id": ref["02_tech.txt"], "grade": 2}],
                },
                {
                    "qid": "q3",
                    "query": "分散データベース 責任者",
                    "relevant": [{"chunk_id": ref["02_tech.txt"], "grade": 1}],
                },
            ],
        )
        out = tmp_path / "bench_out"
        json_path = tmp_path / "report.json"
        rc = bench.run(
            [
                str(tiny_corpus),
                "--configs",
                "c256=chunk:256,overlap:32,wiki:off",
                "--configs",
                "c512=chunk:512,overlap:64,wiki:off",
                "--qrels",
                str(qrels_path),
                "--out",
                str(out),
                "--json",
                str(json_path),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        # 2 config 分の行
        assert "c256" in captured.out
        assert "c512" in captured.out
        # 4 metric 名が header に含まれる (rich は space-separated)
        for metric in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
            assert metric in captured.out
        # JSON report
        assert json_path.exists()
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["schema_version"] == 1
        assert len(report["configs"]) == 2
        for cfg in report["configs"]:
            assert cfg["status"] == "ok"
            assert set(cfg["metrics"].keys()) == set(bench.METRICS)
            # scope-trim: significance column は無い
            assert "significance" not in cfg
            assert "pvalue" not in cfg

    def test_insufficient_configs_exits_2(
        self, tiny_corpus: Path, tmp_path: Path
    ) -> None:
        qrels_path = tmp_path / "q.jsonl"
        _write_qrels(
            qrels_path,
            [{"qid": "q1", "query": "test", "relevant": [{"chunk_id": "x", "grade": 1}]}],
        )
        rc = bench.run(
            [
                str(tiny_corpus),
                "--configs",
                "only=chunk:256,wiki:off",
                "--qrels",
                str(qrels_path),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 2

    def test_empty_qrels_exits_2(
        self, tiny_corpus: Path, tmp_path: Path
    ) -> None:
        qrels_path = tmp_path / "q.jsonl"
        qrels_path.write_text("", encoding="utf-8")
        rc = bench.run(
            [
                str(tiny_corpus),
                "--configs",
                "c256=chunk:256,wiki:off",
                "--configs",
                "c512=chunk:512,wiki:off",
                "--qrels",
                str(qrels_path),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 2

    def test_missing_qrels_exits_2(
        self, tiny_corpus: Path, tmp_path: Path
    ) -> None:
        rc = bench.run(
            [
                str(tiny_corpus),
                "--configs",
                "c256=chunk:256,wiki:off",
                "--configs",
                "c512=chunk:512,wiki:off",
                "--qrels",
                str(tmp_path / "nonexistent.jsonl"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 2

    def test_invalid_config_spec_exits_2(
        self, tiny_corpus: Path, tmp_path: Path
    ) -> None:
        qrels_path = tmp_path / "q.jsonl"
        _write_qrels(
            qrels_path,
            [{"qid": "q1", "query": "test", "relevant": [{"chunk_id": "x", "grade": 1}]}],
        )
        rc = bench.run(
            [
                str(tiny_corpus),
                "--configs",
                "foo=bad",
                "--configs",
                "c512=chunk:512,wiki:off",
                "--qrels",
                str(qrels_path),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 2

    def test_qrels_with_ghost_chunk_ids_metrics_zero_no_crash(
        self, tiny_corpus: Path, tmp_path: Path
    ) -> None:
        # ingest 出力には存在しない chunk_id ばかりの qrels.
        # → ranx は recall=0 で評価し、crash しないこと.
        qrels_path = tmp_path / "q.jsonl"
        _write_qrels(
            qrels_path,
            [
                {"qid": "q1", "query": "合併の背景", "relevant": [{"chunk_id": "000000000000", "grade": 2}]},
                {"qid": "q2", "query": "ベータプロジェクト", "relevant": [{"chunk_id": "111111111111", "grade": 2}]},
            ],
        )
        json_path = tmp_path / "report.json"
        rc = bench.run(
            [
                str(tiny_corpus),
                "--configs",
                "c256=chunk:256,wiki:off",
                "--configs",
                "c512=chunk:512,wiki:off",
                "--qrels",
                str(qrels_path),
                "--out",
                str(tmp_path / "out"),
                "--json",
                str(json_path),
            ]
        )
        assert rc == 0  # partial 成功でも exit 0
        report = json.loads(json_path.read_text(encoding="utf-8"))
        for cfg in report["configs"]:
            assert cfg["status"] == "ok"
            assert cfg["metrics"]["recall@5"] == 0.0
            assert cfg["metrics"]["ndcg@10"] == 0.0


# ---- Subprocess smoke test -------------------------------------------------


class TestBenchSubprocess:
    def test_cli_invocation(self, tiny_corpus: Path, tmp_path: Path) -> None:
        """CLI として subprocess 起動し、exit 0 と JSON の有効性を確認する."""
        ref = _run_ingest_once_for_chunk_ids(tiny_corpus, tmp_path)
        qrels_path = tmp_path / "q.jsonl"
        _write_qrels(
            qrels_path,
            [
                {
                    "qid": "q1",
                    "query": "合併 田中",
                    "relevant": [{"chunk_id": ref["01_news.txt"], "grade": 2}],
                },
                {
                    "qid": "q2",
                    "query": "ベータプロジェクト 分散データベース",
                    "relevant": [{"chunk_id": ref["02_tech.txt"], "grade": 2}],
                },
            ],
        )
        script = _REPO_ROOT / "scripts" / "bench.py"
        json_path = tmp_path / "report.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                str(tiny_corpus),
                "--configs",
                "c256=chunk:256,wiki:off",
                "--configs",
                "c512=chunk:512,wiki:off",
                "--qrels",
                str(qrels_path),
                "--out",
                str(tmp_path / "bench_out"),
                "--json",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
        assert json_path.exists()
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(report["configs"]) == 2


# ---- ranx-missing simulation ----------------------------------------------


class TestRanxMissing:
    def test_ensure_ranx_raises_when_import_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_ensure_ranx` は ImportError を ConfigError にラップする."""
        import builtins
        from lorebook_chunker.errors import ConfigError

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ranx":
                raise ImportError("No module named 'ranx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ConfigError) as exc:
            bench._ensure_ranx()
        assert exc.value.context.get("reason") == "ranx_missing"
        assert "[bench]" in exc.value.context.get("hint", "")
