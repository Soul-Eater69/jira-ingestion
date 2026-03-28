from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional

__all__ = [
    "clean_markdown_text",
    "clean_extracted_text",
    "text_looks_weak",
    "merge_native_and_ocr_text",
]

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_REPEATED_PUNCT_RE = re.compile(r"([\-_=~*•·])\1{3,}")
_ESCAPED_SEQUENCES_RE = re.compile(r"\\(?:r\\n|n|r|t)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")
_DENSE_PIPE_ROW_RE = re.compile(r"\|")
_LONG_SYMBOL_RUN_RE = re.compile(r"[^\w\s]{6,}")
_LITERAL_NULL_RE = re.compile(r"\\x00|\\u0000", re.IGNORECASE)

_BULLET_TRANSLATIONS = {
    ord("•"): "- ",
    ord("◦"): "- ",
    ord("▪"): "- ",
    ord("●"): "- ",
    ord("·"): "- ",
    ord(""): "- ",
    ord("‣"): "- ",
    ord("∙"): "- ",
}

_DASH_TRANSLATIONS = {
    ord("–"): "-",
    ord("—"): "-",
    ord("−"): "-",
}

_QUOTE_TRANSLATIONS = {
    ord("“"): '"',
    ord("”"): '"',
    ord("„"): '"',
    ord("’"): "'",
    ord("‘"): "'",
    ord("`" ): "'",
}


KNOWN_NOISE_PREFIXES = (
    "slide number:",
    "page ",
    "proprietary and confidential",
    "for internal use only",
)


def _unescape_literal_sequences(text: str) -> str:
    if not text or "\\" not in text:
        return text
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\r", "\n")
    text = text.replace("\\t", " ")
    return text


def _undouble_alpha_runs(text: str) -> str:
    """Conservative OCR de-doubling: DDAATTAA -> DATA, not letter -> leter."""
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        chars = list(token)
        if len(chars) >= 4 and len(chars) % 2 == 0 and chars[0::2] == chars[1::2]:
            return "".join(chars[0::2])
        return token

    text = re.sub(r"\b[A-Za-z]{4,}\b", repl, text)
    return re.sub(r"\b([A-Za-z])\1{2,}\b", r"\1", text)


def _despace_spelled_words(text: str) -> str:
    """Collapse D a t a / B r i d g e style OCR output while preserving true spacing."""
    pattern = re.compile(r"(?<!\w)(?:[A-Za-z0-9&]{1,2}\s+){3,}[A-Za-z0-9&]{1,2}(?!\w)")

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        pieces = token.split()
        if all(len(p) <= 2 for p in pieces):
            return "".join(pieces)
        return token

    return pattern.sub(repl, text)


def _drop_sparse_table_rows(text: str) -> str:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if _TABLE_SEP_RE.match(stripped):
            continue
        if _DENSE_PIPE_ROW_RE.search(stripped) and stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            non_empty = [c for c in cells if c]
            if len(cells) >= 3 and (not non_empty or len(non_empty) / len(cells) < 0.4):
                continue
            line = " | ".join(non_empty) if non_empty else ""
        out.append(line)
    return "\n".join(out)


def _strip_noise_lines(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line:
            lines.append("")
            continue
        if _REPEATED_PUNCT_RE.fullmatch(line):
            continue
        if _LONG_SYMBOL_RUN_RE.fullmatch(line):
            continue
        if any(lower.startswith(prefix) for prefix in KNOWN_NOISE_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines)


def _collapse_whitespace_preserve_paragraphs(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_extracted_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = _LITERAL_NULL_RE.sub(" ", text)
    text = _unescape_literal_sequences(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_BULLET_TRANSLATIONS)
    text = text.translate(_DASH_TRANSLATIONS)
    text = text.translate(_QUOTE_TRANSLATIONS)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\xa0", " ").replace("\uf0b7", "- ")
    text = _undouble_alpha_runs(text)
    text = _despace_spelled_words(text)
    text = _drop_sparse_table_rows(text)
    text = _strip_noise_lines(text)
    text = re.sub(r" +([,;:.!?])", r"\1", text)
    text = re.sub(r"\s*([/])\s*", r" \1 ", text)
    text = re.sub(r"[ \t]{3,}", " ", text)
    return _collapse_whitespace_preserve_paragraphs(text)


def clean_markdown_text(text: Optional[str]) -> str:
    return clean_extracted_text(text)


def text_looks_weak(text: Optional[str], min_words: int = 25) -> bool:
    cleaned = clean_extracted_text(text)
    words = cleaned.split()
    if len(words) < min_words:
        return True
    alnum = sum(ch.isalnum() for ch in cleaned)
    ratio = alnum / max(1, len(cleaned))
    line_count = max(1, len(cleaned.splitlines()))
    avg_words_per_line = len(words) / line_count
    return ratio < 0.45 or avg_words_per_line < 2.0


def merge_native_and_ocr_text(native_text: Optional[str], ocr_text: Optional[str]) -> str:
    native = clean_extracted_text(native_text)
    ocr = clean_extracted_text(ocr_text)
    if not native:
        return ocr
    if not ocr:
        return native
    if ocr in native:
        return native
    if native in ocr:
        return ocr
    return f"{native}\n\n[OCR]\n{ocr}"
