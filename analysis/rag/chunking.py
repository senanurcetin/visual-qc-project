"""Split Wikipedia plain-text extracts into section-aware passages."""
from __future__ import annotations

import re
from dataclasses import dataclass

SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes", "bibliography",
    "sources", "citations", "footnotes", "gallery", "notes and references", "literature",
}

_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Passage:
    section: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def strip_math(text: str) -> str:
    """Remove `{\\displaystyle ...}` blocks that the plain-text extract leaves behind for formulas."""
    out, i, marker = [], 0, "{\\displaystyle"
    while True:
        start = text.find(marker, i)
        if start < 0:
            out.append(text[i:])
            break
        out.append(text[i:start])
        depth, j = 0, start
        while j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
            if depth == 0:
                break
        i = j
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


def split_sections(extract: str) -> list[tuple[str, list[str]]]:
    """Return (section title, paragraphs) pairs, dropping reference-style sections."""
    sections: list[tuple[str, list[str]]] = []
    title, paragraphs, skipping = "Introduction", [], False
    top_level = None
    for raw in strip_math(extract).splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _HEADING.match(line)
        if match:
            if paragraphs and not skipping:
                sections.append((title, paragraphs))
            level, name = len(match.group(1)), match.group(2)
            if level == 2:
                top_level = name
                skipping = name.lower() in SKIP_SECTIONS
            elif top_level and top_level.lower() in SKIP_SECTIONS:
                skipping = True
            else:
                skipping = skipping or name.lower() in SKIP_SECTIONS
            title = name if level == 2 or not top_level else f"{top_level} / {name}"
            paragraphs = []
            continue
        if not skipping:
            paragraphs.append(line)
    if paragraphs and not skipping:
        sections.append((title, paragraphs))
    return sections


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_sections(
    sections: list[tuple[str, list[str]]],
    target_words: int = 140,
    max_words: int = 200,
    min_words: int = 40,
) -> list[Passage]:
    """Pack sentences into passages of roughly `target_words`, never crossing a section."""
    passages: list[Passage] = []
    for title, paragraphs in sections:
        buffer: list[str] = []
        count = 0
        section_passages: list[Passage] = []
        for sentence in (s for p in paragraphs for s in split_sentences(p)):
            words = len(sentence.split())
            if buffer and (count + words > max_words or count >= target_words):
                section_passages.append(Passage(title, " ".join(buffer)))
                buffer, count = [], 0
            buffer.append(sentence)
            count += words
        if buffer:
            tail = Passage(title, " ".join(buffer))
            if section_passages and tail.word_count < min_words:
                last = section_passages[-1]
                section_passages[-1] = Passage(title, f"{last.text} {tail.text}")
            else:
                section_passages.append(tail)
        passages.extend(p for p in section_passages if p.word_count >= min_words // 2)
    return passages
