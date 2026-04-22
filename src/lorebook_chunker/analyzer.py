"""JapaneseAnalyzer: Ginza ラッパ + 正規化 + analyzer.json I/O.

設計要点 (plan の KTD に従う):
- `token._.ne` 経由で OntoNotes5 NER ラベルを取得 (`doc.ents` は使用しない)
- strict_match (tokenization を決定的に変える) と compat_match (major.minor 一致) を分離
- Sudachi dict は binary SHA-256 を strict_match に含める
"""
from __future__ import annotations

import bisect
import functools
import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from lorebook_chunker.normalize import NORMALIZATION_SPEC, normalize_text
from lorebook_chunker.schema import (
    AnalyzerConfig,
    AnalyzerNEUnavailableError,
    AnalyzerVersionMismatchError,
    EntityMention,
)

# SudachiPy の 1 回の入力長上限は 49149 バイト (CHAR_DEF.len 由来). spaCy + GiNZA 経由
# だと Doc 構築時に全文を一括 tokenize するため、長文を渡すと SudachiError で落ちる.
# そこで analyzer 側で段落境界に沿って事前分割し、各セグメントを個別に _nlp() に
# 通す. 閾値はハード上限に対する安全マージン (将来 spaCy/Sudachi 側で境界計算オーバヘッドが
# 増えた時の耐性) を確保するため保守的に取った.
SUDACHI_INPUT_BYTE_LIMIT = 40000

DEFAULT_POS_ALLOWLIST = ("NOUN", "VERB", "ADJ", "PROPN")
DEFAULT_STOPWORDS: tuple[str, ...] = (
    "こと",
    "もの",
    "ため",
    "よう",
    "ところ",
    "これ",
    "それ",
    "あれ",
    "いる",
    "ある",
    "する",
    "なる",
    "くる",
)
DEFAULT_SPLIT_MODE = "C"
DEFAULT_MODEL_NAME = "ja_ginza_electra"


def _split_into_nlp_segments(
    text: str, byte_limit: int = SUDACHI_INPUT_BYTE_LIMIT
) -> list[tuple[int, str]]:
    """byte_limit 以下の UTF-8 バイト長に収まるセグメントに分割.

    返り値は (char_offset, segment_text) のリスト. segment_text を順に結合すると
    元の text と完全一致する (concatenation invariant — char offset 保存のため必須).
    入力が byte_limit 以下なら [(0, text)] を返す no-op.

    分割優先度: 段落境界 (``\\n\\n``) → 文末 (``。``) → 改行 (``\\n``) → ハードカット.
    いずれも切れ目が見つからない場合は byte_limit に収まる最大文字数でそのまま切る.

    高速化: 旧実装は二分探索の各ステップで ``text[:mid].encode('utf-8')`` を呼び
    O(N^2 log N) のコピー + encoding を行っていた. 現実装は char→byte 累積バイト数
    配列を 1 度だけ作り (O(N)), bisect でカット位置を O(log N) で解決する.
    """
    if not text:
        return []
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return [(0, text)]

    # 累積バイト長: cumu[i] = text[:i] の UTF-8 バイト長.
    # UTF-8 は 1 文字 1〜4 バイトなので ord(c) から byte 数を得る.
    cumu: list[int] = [0] * (len(text) + 1)
    acc = 0
    for i, ch in enumerate(text):
        code = ord(ch)
        if code < 0x80:
            acc += 1
        elif code < 0x800:
            acc += 2
        elif 0xD800 <= code <= 0xDFFF:  # surrogate (shouldn't happen in str, defensive)
            acc += 3  # pragma: no cover
        elif code < 0x10000:
            acc += 3
        else:
            acc += 4
        cumu[i + 1] = acc

    segments: list[tuple[int, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        # tail のバイト長 = cumu[n] - cumu[pos]
        if cumu[n] - cumu[pos] <= byte_limit:
            segments.append((pos, text[pos:]))
            break
        # byte_limit に収まる最大 char index を bisect で求める.
        # 条件: cumu[pos + k] - cumu[pos] <= byte_limit となる最大 k.
        target = cumu[pos] + byte_limit
        # bisect_right で "cumu 値 <= target" を満たす最右 index を得る.
        hi_idx = bisect.bisect_right(cumu, target, lo=pos + 1, hi=n + 1) - 1
        max_chars = hi_idx - pos
        if max_chars <= 0:
            max_chars = 1  # 1 文字が byte_limit を超えるケースの安全弁
        window = text[pos:pos + max_chars]
        # 段落 → 文末 → 改行 → ハードカット の順で切れ目を探す
        cut = window.rfind("\n\n")
        if cut >= 0:
            cut += 2
        else:
            cut = window.rfind("。")
            if cut >= 0:
                cut += 1
            else:
                cut = window.rfind("\n")
                if cut >= 0:
                    cut += 1
                else:
                    cut = max_chars
        if cut <= 0:
            cut = max_chars
        segments.append((pos, text[pos:pos + cut]))
        pos += cut
    return segments


@functools.lru_cache(maxsize=1)
def _sudachidict_binary_sha256() -> str:
    """system.dic の SHA-256 を返す. dict 実体を検知するための strict_match フィールド.

    dict binary はプロセス寿命内で不変なので lru_cache でメモ化する
    (build_config / compute_analyzer_json_hash / save が 1 ingest あたり 2〜3 回
    呼び出すため、毎回 ~1.5s かかっていた I/O を 1 回分に圧縮する).
    """
    try:
        import sudachidict_core  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - environment-dependent
        return "sudachidict_core-not-available"
    pkg_dir = Path(sudachidict_core.__file__).parent
    dic = pkg_dir / "resources" / "system.dic"
    if not dic.exists():
        return "system.dic-not-found"
    h = hashlib.sha256()
    with dic.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


try:
    from ginza import ENE_ONTONOTES_MAPPING as _ENE_ONTONOTES_MAPPING  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - ginza pinned in pyproject
    _ENE_ONTONOTES_MAPPING = {}


def _ne_getter(token: Any) -> str | None:
    """Token._.ne の getter. `B-ORG` / `I-PERSON` / None を派生させる.

    nlp.pipe(n_process=N) でワーカーを spawn すると Language オブジェクトが pickle
    される. Token.set_extension の getter はこの pickle ルートで辿られるため、
    クロージャではなくモジュールトップレベル関数でないと AttributeError になる.
    """
    ene = token.ent_type_
    iob = token.ent_iob_
    if not ene or iob in ("", "O"):
        return None
    onto = _ENE_ONTONOTES_MAPPING.get(ene, ene)
    return f"{iob}-{onto}"


def _model_checksum(nlp: Any) -> str:
    """spaCy Language オブジェクトのメタ情報から安定したチェックサムを取り出す.

    正確な weights hash が取れない場合は meta JSON を sha256 する fallback。
    """
    meta = getattr(nlp, "meta", None) or {}
    payload = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "meta:" + hashlib.sha256(payload).hexdigest()[:16]


class JapaneseAnalyzer:
    """Ginza + Sudachi ベースの日本語テキスト解析ラッパ."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        split_mode: str = DEFAULT_SPLIT_MODE,
        pos_allowlist: Iterable[str] = DEFAULT_POS_ALLOWLIST,
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
        tfidf_config: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        import spacy  # lazy import

        self._model_name = model_name
        self._split_mode = split_mode
        self._pos_allowlist = tuple(pos_allowlist)
        self._stopwords = frozenset(stopwords)
        self._tfidf_config = tfidf_config or {"min_df": 1, "max_df": 0.95}
        self._device = device
        # Ginza 5.2 の compound_splitter は split_mode を None 既定で登録するため
        # spacy 3.8 の厳格な Config 検証に失敗する. 明示的に上書きして型エラーを回避.
        # また bunsetu_recognizer は本 CLI のどの出力にも使われていない上、長文や
        # 語彙連結コーパスで再帰的に依存グラフを走査して RecursionError を起こすため
        # exclude で load 時点から除外する (load コスト + 推論コスト両方を削減).
        self._nlp = spacy.load(
            model_name,
            exclude=["bunsetu_recognizer"],
            config={"components": {"compound_splitter": {"split_mode": split_mode}}},
        )
        # Optional: move transformer (ELECTRA) to non-CPU device for inference speed.
        # ja_ginza (non-transformer) では transformer pipe が無いので warning のみ.
        if device != "cpu":
            self._move_to_device(device)
        # ja_ginza_electra 5.2 では token._.ne が自動登録されないため、本プロジェクト側で
        # doc.ents + ENE_ONTONOTES_MAPPING から BIO + OntoNotes5 ラベルを派生させた上で
        # Token 拡張 "ne" を登録する (plan の BIO walk ロジックはそのまま使える).
        self._register_ne_extension()
        self._verify_ne_extension()

    def _move_to_device(self, device: str) -> None:
        """transformer pipe を指定デバイスに移す (MPS / CUDA 想定).

        - ja_ginza_electra の場合: `nlp.get_pipe('transformer').model.transformer` が
          HuggingFace ElectraModel (PreTrainedModel). `.to(device)` で移動可能.
        - ja_ginza の場合: transformer pipe が無いので no-op + RuntimeWarning.
        - MPS: spacy が native には対応していないが、PyTorch 層を直接 MPS に移せば
          inference は通る (確認済み). 数値は CPU と bit-exact ではないので POS/NER の
          argmax 境界で結果が揺れる可能性あり (vocab.npz に影響することに注意).
        """
        import warnings

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch は spacy の推移依存
            raise RuntimeError(
                f"device={device!r} が要求されましたが torch が import できません: {exc}"
            ) from exc

        if device == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                warnings.warn(
                    "device='mps' が要求されましたがこの環境で MPS が使えないため "
                    "CPU のまま継続します.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
        elif device == "cuda":
            if not torch.cuda.is_available():
                warnings.warn(
                    "device='cuda' が要求されましたが CUDA が使えないため "
                    "CPU のまま継続します.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
        else:
            raise ValueError(f"unknown device: {device!r} (expected cpu/mps/cuda)")

        torch_device = torch.device(device)
        try:
            transformer_pipe = self._nlp.get_pipe("transformer")
        except KeyError:
            warnings.warn(
                f"model={self._model_name!r} は transformer pipe を持たないため "
                f"device={device!r} への移動を skip します (tok2vec は CPU で動作).",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        inner = getattr(transformer_pipe.model, "transformer", None)
        if inner is None or not hasattr(inner, "to"):
            warnings.warn(
                "transformer pipe の内部 PyTorch model が見つからず device 移動できません.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        inner.to(torch_device)

    @staticmethod
    def _register_ne_extension() -> None:
        from spacy.tokens import Token  # lazy import

        if Token.has_extension("ne"):
            return
        # getter はモジュールトップレベルの _ne_getter を参照. クロージャにすると
        # nlp.pipe(n_process>1) の spawn で pickle できず AttributeError で落ちる.
        Token.set_extension("ne", getter=_ne_getter)

    # ---- NER availability check --------------------------------------

    def _verify_ne_extension(self) -> None:
        """token._.ne 拡張 + NER コンポーネントの両方が揃っていることを確認.

        以前は "スカラー商事は東京で発表した。" を実際に ELECTRA に通して B-* タグを
        検証していたが、これは ingest 起動時 + multiprocessing ワーカー起動時ごとに
        300〜500 ms 分の transformer inference を誘発していた. 拡張登録と pipe 存在
        だけを静的に検査すれば、本 CLI が想定する失敗モード (extension 未登録 /
        NER 欠落) は同じ粒度で検出できる.
        """
        from spacy.tokens import Token

        if not Token.has_extension("ne"):
            raise AnalyzerNEUnavailableError(
                "token._.ne 拡張が未登録です (register_ne_extension 前に verify した?)"
            )
        pipe_names = set(getattr(self._nlp, "pipe_names", ()) or ())
        if "ner" not in pipe_names:
            raise AnalyzerNEUnavailableError(
                f"Ginza モデル {self._model_name!r} に NER コンポーネントがありません "
                f"(pipe_names={sorted(pipe_names)})."
            )

    # ---- Pipeline operations -----------------------------------------

    # ---- Doc-level primitives (batch / reuse フレンドリ) ---------------
    #
    # これらは ingest ホットパスから chunk ごとの Doc を 1 回だけ計算した後、
    # TF-IDF 側の token 抽出と NER 側の span 抽出の両方で使い回す用.
    # 元の iter_* / tokenize_for_tfidf は query 側互換のため残し、内部で
    # これらを呼び出す thin wrapper とする.

    def nlp_pipe(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 32,
        n_process: int = 1,
    ) -> Iterator[Any]:
        """spaCy の batch 推論 wrapper. 各 text は SUDACHI_INPUT_BYTE_LIMIT 以下を前提.

        ingest の chunk 単位 (~500 字) で呼ぶことを想定. 長文を流す場合は呼び出し側で
        `_split_into_nlp_segments` で分割してから渡すこと.
        ``n_process`` > 1 で multiprocessing 並列 (spaCy native). 初回 worker 起動時に
        各プロセスで ELECTRA を load するコスト (~数秒) があるので、短い入力では
        n_process=1 の方が速い場合がある.
        """
        return self._nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

    def batch_iter_sentences(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        n_process: int = 1,
        progress: Any = None,
        progress_label: str = "文境界抽出",
    ) -> list[list[str]]:
        """複数の長文テキストの文境界抽出を 1 回の nlp.pipe に束ねて実行.

        各 text を `_split_into_nlp_segments` で分割し、全 text の全セグメントを
        1 本の nlp.pipe に流す. 返り値は texts と同長で、各要素は対応 text の
        文字列リスト (doc.sents.text の flatten).

        chunker の iter_sentences を 1 ファイルずつ呼ぶ逐次ループをまとめて
        batch + 並列に潰すために使う.

        ``progress`` が与えられれば ``progress.start(label, total=segments) →
        progress.tick() → progress.end()`` の lifecycle を本メソッドが管理する.
        """
        if not texts:
            return []
        # segments を flatten しつつ、どの text に属するかのオーナーを記録
        segments_flat: list[str] = []
        owner: list[int] = []
        for i, text in enumerate(texts):
            if not text:
                continue
            for _off, seg in _split_into_nlp_segments(text):
                segments_flat.append(seg)
                owner.append(i)
        sents_per_text: list[list[str]] = [[] for _ in texts]
        if not segments_flat:
            return sents_per_text
        if progress is not None:
            progress.start(progress_label, total=len(segments_flat))
        try:
            for idx, doc in enumerate(
                self._nlp.pipe(segments_flat, batch_size=batch_size, n_process=n_process)
            ):
                text_idx = owner[idx]
                for sent in doc.sents:
                    if sent.text:
                        sents_per_text[text_idx].append(sent.text)
                if progress is not None:
                    progress.tick()
        finally:
            if progress is not None:
                progress.end()
        return sents_per_text

    def tokens_from_doc(self, doc: Any) -> list[str]:
        """事前計算した Doc から TF-IDF 用トークン列を取り出す (POS + lemma + stopword)."""
        allow = self._pos_allowlist
        stop = self._stopwords
        out: list[str] = []
        for token in doc:
            if token.pos_ not in allow:
                continue
            lemma = token.lemma_.strip()
            if not lemma or lemma in stop:
                continue
            out.append(lemma)
        return out

    # ---- Single-pass document analysis (ingest ホットパスの主役) -----
    #
    # analyze_documents は ingest が欲しい情報 (文境界・TF-IDF 用 lemma 列・
    # 絶対 char offset 付き entity 列) をファイル単位で 1 回の nlp.pipe で
    # 揃える. 旧実装は
    #   (a) batch_iter_sentences で全文に ELECTRA を掛け文境界を取り、
    #   (b) _compute_tokens_and_entities_via_pipe で同じ内容のチャンクに
    #       もう一度 ELECTRA を掛けて tokens と entities を取る,
    # という 2 パス構成で、同一コーパスを ELECTRA に 2 回通していた.
    # analyze_documents は 1 パスで済むため ingest の NLP 時間を半減させる.

    def analyze_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        n_process: int = 1,
        progress: Any = None,
        progress_label: str = "文書解析 (nlp.pipe)",
    ) -> list["DocumentAnalysis"]:
        """長文テキスト列をまとめて 1 回の nlp.pipe で解析し、
        ``DocumentAnalysis`` を返す (sentences + tf-idf tokens + absolute-offset entities).

        各 text は ``_split_into_nlp_segments`` で SUDACHI 上限に沿って分割し、
        全 text の全セグメントを 1 本の nlp.pipe に流して並列計算する. token の
        char offset / 文字列長は ``token.idx + segment_offset`` で絶対化する.
        """
        if not texts:
            return []
        # segments_flat: nlp.pipe に流す実体. owner: セグメントが属する text idx.
        # offsets: そのセグメントの text 内 char offset.
        segments_flat: list[str] = []
        owners: list[int] = []
        offsets: list[int] = []
        for ti, text in enumerate(texts):
            if not text:
                continue
            for off, seg in _split_into_nlp_segments(text):
                segments_flat.append(seg)
                owners.append(ti)
                offsets.append(off)
        builders: list[_DocumentAnalysisBuilder] = [
            _DocumentAnalysisBuilder() for _ in texts
        ]
        if not segments_flat:
            return [b.build() for b in builders]

        allow = self._pos_allowlist
        stop = self._stopwords

        if progress is not None:
            progress.start(progress_label, total=len(segments_flat))
        try:
            for idx, doc in enumerate(
                self._nlp.pipe(
                    segments_flat, batch_size=batch_size, n_process=n_process
                )
            ):
                ti = owners[idx]
                off = offsets[idx]
                b = builders[ti]
                # Sentences (文境界は chunker が使う)
                for sent in doc.sents:
                    if sent.text:
                        b.sentences.append(sent.text)
                # TF-IDF lemma tokens (POS + stopword フィルタ込み, 絶対 char offset)
                starts = b.tfidf_token_starts
                lemmas = b.tfidf_token_lemmas
                for token in doc:
                    if token.pos_ not in allow:
                        continue
                    lemma = token.lemma_.strip()
                    if not lemma or lemma in stop:
                        continue
                    starts.append(token.idx + off)
                    lemmas.append(lemma)
                # Entities (絶対 char offset)
                for ent in self._walk_bio_spans(doc, char_offset=off):
                    b.entities.append(ent)
                if progress is not None:
                    progress.tick()
        finally:
            if progress is not None:
                progress.end()
        return [b.build() for b in builders]

    def entities_from_doc(
        self, doc: Any, char_offset: int = 0
    ) -> Iterator[EntityMention]:
        """事前計算した Doc から BIO span を構築して EntityMention を yield."""
        return self._walk_bio_spans(doc, char_offset=char_offset)

    # ---- Text-level convenience APIs (query 側互換) --------------------

    def iter_sentences(self, text: str) -> Iterator[str]:
        """文境界で分割. 入力はすでに normalize_text 済み前提.

        長文 (> SUDACHI_INPUT_BYTE_LIMIT) は段落境界で事前分割して各セグメントを
        順に _nlp() に通す. 段落境界は文境界の上位なので、境界跨ぎによる文切れは
        発生しない.
        """
        if not text:
            return iter([])

        def _gen() -> Iterator[str]:
            for _offset, segment in _split_into_nlp_segments(text):
                doc = self._nlp(segment)
                for sent in doc.sents:
                    if sent.text:
                        yield sent.text

        return _gen()

    def iter_entities(self, text: str) -> Iterator[EntityMention]:
        """text → Doc → EntityMention の thin wrapper.

        長文はセグメント分割し、各セグメントの char_offset を global offset に
        復元して EntityMention を yield する. セグメント境界跨ぎのエンティティは
        失う可能性があるが、分割は段落境界優先なので実害は小さい.
        """
        if not text:
            return iter([])

        def _gen() -> Iterator[EntityMention]:
            for offset, segment in _split_into_nlp_segments(text):
                doc = self._nlp(segment)
                yield from self.entities_from_doc(doc, char_offset=offset)

        return _gen()

    def _walk_bio_spans(
        self, doc: Any, char_offset: int = 0
    ) -> Iterator[EntityMention]:
        current_label: str | None = None
        start_char: int | None = None
        end_char: int | None = None
        for token in doc:
            tag = getattr(token._, "ne", None) or "O"
            if tag.startswith("B-"):
                if current_label and start_char is not None and end_char is not None:
                    yield self._emit(doc.text, current_label, start_char, end_char, char_offset)
                current_label = tag[2:]
                start_char = token.idx
                end_char = token.idx + len(token.text)
            elif tag.startswith("I-") and current_label == tag[2:] and start_char is not None:
                end_char = token.idx + len(token.text)
            else:
                if current_label and start_char is not None and end_char is not None:
                    yield self._emit(doc.text, current_label, start_char, end_char, char_offset)
                current_label = None
                start_char = None
                end_char = None
        if current_label and start_char is not None and end_char is not None:
            yield self._emit(doc.text, current_label, start_char, end_char, char_offset)

    @staticmethod
    def _emit(
        text: str, label: str, start: int, end: int, char_offset: int = 0
    ) -> EntityMention:
        return EntityMention(
            name=text[start:end],
            ner_label=label,
            char_start=start + char_offset,
            char_end=end + char_offset,
        )

    def tokenize_for_tfidf(self, text: str) -> list[str]:
        """text → Doc → tokens の thin wrapper. query 側で使う.

        通常はチャンク単位 (~500 字) で呼ばれるため no-op だが、全文で呼ばれた
        場合も同じ段落分割ロジックで長文対応する.
        """
        if not text:
            return []
        out: list[str] = []
        for _offset, segment in _split_into_nlp_segments(text):
            out.extend(self.tokens_from_doc(self._nlp(segment)))
        return out

    # ---- analyzer.json I/O -------------------------------------------

    def build_config(self) -> AnalyzerConfig:
        strict = {
            "model_name": self._model_name,
            "model_checksum": _model_checksum(self._nlp),
            "split_mode": self._split_mode,
            "pos_allowlist": list(self._pos_allowlist),
            "stopwords": sorted(self._stopwords),
            "lemma_rules": "lemma_ field as-is",
            "sudachidict_binary_sha256": _sudachidict_binary_sha256(),
            "normalization": dict(NORMALIZATION_SPEC),
        }
        compat = {
            "ginza": _package_version("ginza"),
            "spacy": _package_version("spacy"),
            "sudachipy": _package_version("sudachipy"),
            "sudachidict_package": "sudachidict_core",
            "sudachidict_package_version": _package_version("sudachidict_core"),
        }
        return AnalyzerConfig(
            strict_match=strict,
            compat_match=compat,
            tfidf=dict(self._tfidf_config),
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """analyzer.json を atomic write (F-057): tempfile + fsync + os.replace."""
        config = self.build_config()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".analyzer.", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load_and_verify(cls, path: str | os.PathLike[str]) -> "JapaneseAnalyzer":
        """analyzer.json を読み、保存された model_name/split_mode で analyzer を構築.

        意図: ingest 時と **同一の** ランタイム設定で query / lint / 再 ingest が
        実行されていることを保証する. 以前は `cls()` で defaults で復元していたが、
        これだと --analyzer-backend ginza で ingest したあと defaults (electra) で
        query しようとすると strict_match 以前に import error / 別モデル load になる.
        現在は保存された model_name を復元して analyzer を構築し、strict_match の
        全キーを突合する (model_name / split_mode も diff 対象に含まれる).
        device は strict_match に含まれないため、query 側は常に CPU で動作する.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            saved = AnalyzerConfig.from_dict(json.load(f))
        saved_model = saved.strict_match.get("model_name", DEFAULT_MODEL_NAME)
        # model_name だけ saved から取り、analyzer を構築する. split_mode 等は
        # defaults (現ランタイム) で構築して strict_match 比較に渡すことで、
        # 改ざんや環境ドリフト (例: split_mode を書き換え) を検知する.
        analyzer = cls(model_name=saved_model)
        current = analyzer.build_config()
        diffs = _strict_diff(saved.strict_match, current.strict_match)
        if diffs:
            raise AnalyzerVersionMismatchError(
                "analyzer.json の strict_match が現在のランタイムと不一致: "
                + ", ".join(diffs)
            )
        _warn_compat_mismatch(saved.compat_match, current.compat_match)
        return analyzer


def _strict_diff(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    keys = set(saved) | set(current)
    for key in sorted(keys):
        if saved.get(key) != current.get(key):
            diffs.append(key)
    return diffs


def _warn_compat_mismatch(saved: dict[str, Any], current: dict[str, Any]) -> None:
    import warnings

    for key in ("ginza", "spacy", "sudachipy", "sudachidict_package_version"):
        sv = str(saved.get(key, ""))
        cv = str(current.get(key, ""))
        if sv != cv and _major_minor(sv) != _major_minor(cv):
            warnings.warn(
                f"analyzer.json の {key!r} が不一致: 保存時 {sv!r}, 現在 {cv!r}. "
                "major.minor が揃っていないため挙動が変わる可能性があります.",
                RuntimeWarning,
                stacklevel=2,
            )


def _major_minor(version: str) -> str:
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


# ---- single-pass document analysis containers ---------------------------


@dataclass
class _DocumentAnalysisBuilder:
    """analyze_documents が 1 セグメントずつ蓄積するためのミュータブル構造."""

    sentences: list[str] = field(default_factory=list)
    tfidf_token_starts: list[int] = field(default_factory=list)
    tfidf_token_lemmas: list[str] = field(default_factory=list)
    entities: list[EntityMention] = field(default_factory=list)

    def build(self) -> "DocumentAnalysis":
        return DocumentAnalysis(
            sentences=self.sentences,
            tfidf_token_starts=self.tfidf_token_starts,
            tfidf_token_lemmas=self.tfidf_token_lemmas,
            entities=self.entities,
        )


@dataclass
class DocumentAnalysis:
    """1 ファイル分の単一パス解析結果.

    TF-IDF 用 lemma と entity は **絶対 char offset** (ファイル先頭からのオフセット)
    を保持する. chunker が決めたチャンク範囲 [start, end) に対して bisect で
    スライスし、chunk 単位に変換する.
    """

    sentences: list[str]
    tfidf_token_starts: list[int]  # 各要素は絶対 char offset (昇順)
    tfidf_token_lemmas: list[str]  # starts と同長の parallel list
    entities: list[EntityMention]  # 絶対 char offset

    def tokens_in_range(self, start: int, end: int) -> list[str]:
        """[start, end) の範囲に入る TF-IDF lemma を返す.

        `token.idx + seg_offset` で格納しているので ``bisect`` で端点を解決できる.
        """
        lo = bisect.bisect_left(self.tfidf_token_starts, start)
        hi = bisect.bisect_left(self.tfidf_token_starts, end)
        if lo == 0 and hi == len(self.tfidf_token_lemmas):
            return list(self.tfidf_token_lemmas)
        return self.tfidf_token_lemmas[lo:hi]

    def entities_in_range(
        self, start: int, end: int, target_labels: frozenset[str] | set[str]
    ) -> list[EntityMention]:
        """[start, end) の範囲に完全に収まる entity を chunk-relative offset で返す.

        チャンク境界をまたぐ span は除外する (呼び出し側のコントラクト: ChunkRecord.entities
        は chunk 内相対オフセット).
        """
        out: list[EntityMention] = []
        for m in self.entities:
            if m.char_start < start or m.char_end > end:
                continue
            if m.ner_label not in target_labels:
                continue
            out.append(
                EntityMention(
                    name=m.name,
                    ner_label=m.ner_label,
                    char_start=m.char_start - start,
                    char_end=m.char_end - start,
                )
            )
        return out


# ---- text normalization passthrough (convenience export) -----------------

__all__ = [
    "JapaneseAnalyzer",
    "DocumentAnalysis",
    "normalize_text",
    "DEFAULT_POS_ALLOWLIST",
    "DEFAULT_STOPWORDS",
    "DEFAULT_SPLIT_MODE",
    "DEFAULT_MODEL_NAME",
]
