"""JapaneseAnalyzer: Ginza ラッパ + 正規化 + analyzer.json I/O.

設計要点 (plan の KTD に従う):
- `token._.ne` 経由で OntoNotes5 NER ラベルを取得 (`doc.ents` は使用しない)
- strict_match (tokenization を決定的に変える) と compat_match (major.minor 一致) を分離
- Sudachi dict は binary SHA-256 を strict_match に含める
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from lorebook_chunker.normalize import NORMALIZATION_SPEC, normalize_text
from lorebook_chunker.schema import (
    AnalyzerConfig,
    AnalyzerNEUnavailableError,
    AnalyzerVersionMismatchError,
    EntityMention,
)

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


def _sudachidict_binary_sha256() -> str:
    """system.dic の SHA-256 を返す. dict 実体を検知するための strict_match フィールド."""
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
    ) -> None:
        import spacy  # lazy import

        self._model_name = model_name
        self._split_mode = split_mode
        self._pos_allowlist = tuple(pos_allowlist)
        self._stopwords = frozenset(stopwords)
        self._tfidf_config = tfidf_config or {"min_df": 1, "max_df": 0.95}
        # Ginza 5.2 の compound_splitter は split_mode を None 既定で登録するため
        # spacy 3.8 の厳格な Config 検証に失敗する. 明示的に上書きして型エラーを回避.
        self._nlp = spacy.load(
            model_name,
            config={"components": {"compound_splitter": {"split_mode": split_mode}}},
        )
        # ja_ginza_electra 5.2 では token._.ne が自動登録されないため、本プロジェクト側で
        # doc.ents + ENE_ONTONOTES_MAPPING から BIO + OntoNotes5 ラベルを派生させた上で
        # Token 拡張 "ne" を登録する (plan の BIO walk ロジックはそのまま使える).
        self._register_ne_extension()
        self._verify_ne_extension()

    @staticmethod
    def _register_ne_extension() -> None:
        from spacy.tokens import Token  # lazy import

        if Token.has_extension("ne"):
            return
        try:
            from ginza import ENE_ONTONOTES_MAPPING  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover - ginza pinned in pyproject
            ENE_ONTONOTES_MAPPING = {}

        def _ne_getter(token: Any) -> str | None:
            ene = token.ent_type_
            iob = token.ent_iob_
            if not ene or iob in ("", "O"):
                return None
            onto = ENE_ONTONOTES_MAPPING.get(ene, ene)
            return f"{iob}-{onto}"

        Token.set_extension("ne", getter=_ne_getter)

    # ---- NER availability check --------------------------------------

    def _verify_ne_extension(self) -> None:
        """token._.ne が利用可能で、かつ既知のエンティティで populate されることを確認.

        既知エンティティ入り文を流して最低 1 件の B-* タグが得られなければ raise。
        (extension が未登録のパイプライン / NER コンポーネント欠落を検出する意図)
        """
        try:
            doc = self._nlp("スカラー商事は東京で発表した。")
        except Exception as exc:  # pragma: no cover
            raise AnalyzerNEUnavailableError(
                f"Ginza パイプラインが初期化できません: {exc}"
            ) from exc
        try:
            tags = [getattr(t._, "ne", None) for t in doc]
        except AttributeError as exc:
            raise AnalyzerNEUnavailableError(
                f"token._.ne 拡張が未登録: {exc}"
            ) from exc
        if not any(isinstance(t, str) and t.startswith("B-") for t in tags):
            raise AnalyzerNEUnavailableError(
                f"token._.ne が populate されていません。Ginza モデル {self._model_name!r} の "
                "NER コンポーネントが有効か確認してください。"
            )

    # ---- Pipeline operations -----------------------------------------

    def iter_sentences(self, text: str) -> Iterator[str]:
        """文境界で分割. 入力はすでに normalize_text 済み前提."""
        if not text:
            return iter([])
        doc = self._nlp(text)
        return (sent.text for sent in doc.sents if sent.text)

    def iter_entities(self, text: str) -> Iterator[EntityMention]:
        """token._.ne BIO タグを走査して span を構築.

        `B-XXX` で開始、連続 `I-XXX` を併合、`O` / 別 `B-` / 末尾で終端。
        """
        if not text:
            return iter([])
        doc = self._nlp(text)
        return self._walk_bio_spans(doc)

    def _walk_bio_spans(self, doc: Any) -> Iterator[EntityMention]:
        current_label: str | None = None
        start_char: int | None = None
        end_char: int | None = None
        for token in doc:
            tag = getattr(token._, "ne", None) or "O"
            if tag.startswith("B-"):
                if current_label and start_char is not None and end_char is not None:
                    yield self._emit(doc.text, current_label, start_char, end_char)
                current_label = tag[2:]
                start_char = token.idx
                end_char = token.idx + len(token.text)
            elif tag.startswith("I-") and current_label == tag[2:] and start_char is not None:
                end_char = token.idx + len(token.text)
            else:
                if current_label and start_char is not None and end_char is not None:
                    yield self._emit(doc.text, current_label, start_char, end_char)
                current_label = None
                start_char = None
                end_char = None
        if current_label and start_char is not None and end_char is not None:
            yield self._emit(doc.text, current_label, start_char, end_char)

    @staticmethod
    def _emit(text: str, label: str, start: int, end: int) -> EntityMention:
        return EntityMention(
            name=text[start:end],
            ner_label=label,
            char_start=start,
            char_end=end,
        )

    def tokenize_for_tfidf(self, text: str) -> list[str]:
        """POS フィルタ + lemma 化. TfidfVectorizer の analyzer 引数として使う."""
        if not text:
            return []
        doc = self._nlp(text)
        out: list[str] = []
        for token in doc:
            if token.pos_ not in self._pos_allowlist:
                continue
            lemma = token.lemma_.strip()
            if not lemma:
                continue
            if lemma in self._stopwords:
                continue
            out.append(lemma)
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
        """analyzer.json を読み、現在のランタイム (defaults) と strict_match で突合.

        意図: ingest 時と **同一の** ランタイム設定で query / lint / 再 ingest が
        実行されていることを保証する. 従来実装は保存値をそのまま analyzer に
        渡し返して build_config と比較していたため恒真 (改ざんや環境ドリフト
        を検知できない) だった. 現在は現ランタイムの defaults で analyzer を
        構築し、保存された strict_match と差分があれば raise する.
        将来的にランタイム設定の上書き (config file 等) を足す場合、そこで
        構築済みの analyzer を引数に取るバリアントを追加すること.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            saved = AnalyzerConfig.from_dict(json.load(f))
        analyzer = cls()  # defaults — 現在のランタイム状態を代表する
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


# ---- text normalization passthrough (convenience export) -----------------

__all__ = [
    "JapaneseAnalyzer",
    "normalize_text",
    "DEFAULT_POS_ALLOWLIST",
    "DEFAULT_STOPWORDS",
    "DEFAULT_SPLIT_MODE",
    "DEFAULT_MODEL_NAME",
]
