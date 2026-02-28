"""Text chunking with overlap for context continuity."""

import re
from collections.abc import Iterator
from typing import Any

# Approximate tokens per character (conservative estimate for English)
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 500
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # ~2000 chars
OVERLAP_TOKENS = 50
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~200 chars


def chunk_text(
    text: str,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> Iterator[str]:
    """
    Split text into chunks with overlap.

    Strategy:
    1. Split by paragraphs (double newlines)
    2. Accumulate paragraphs until target size
    3. Include overlap from previous chunk
    """
    if not text.strip():
        return

    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return

    current_chunk: list[str] = []
    current_size = 0
    overlap_text = ""

    for para in paragraphs:
        para_size = len(para)

        # If single paragraph exceeds target, split it by sentences
        if para_size > target_chars:
            # Flush current chunk first
            if current_chunk:
                chunk_text_out = "\n\n".join(current_chunk)
                if overlap_text:
                    chunk_text_out = overlap_text + "\n\n" + chunk_text_out
                yield chunk_text_out.strip()
                overlap_text = "\n\n".join(current_chunk)[-overlap_chars:] if current_chunk else ""
                current_chunk = []
                current_size = 0

            # Split long paragraph by sentences, prepending overlap so context carries through
            para_to_split = (overlap_text + "\n\n" + para) if overlap_text else para
            yield from _split_long_paragraph(para_to_split, target_chars, overlap_chars)
            # Update overlap_text from the end of the original paragraph (not the prepended version)
            overlap_text = para[-overlap_chars:]
            continue

        # Check if adding this paragraph exceeds target
        if current_size + para_size > target_chars and current_chunk:
            # Output current chunk
            chunk_text_out = "\n\n".join(current_chunk)
            if overlap_text:
                chunk_text_out = overlap_text + "\n\n" + chunk_text_out
            yield chunk_text_out.strip()

            # Keep overlap from end of current chunk
            overlap_text = "\n\n".join(current_chunk)[-overlap_chars:] if current_chunk else ""
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += para_size

    # Output remaining content
    if current_chunk:
        chunk_text_out = "\n\n".join(current_chunk)
        if overlap_text:
            chunk_text_out = overlap_text + "\n\n" + chunk_text_out
        yield chunk_text_out.strip()


def _split_long_paragraph(
    text: str,
    target_chars: int,
    overlap_chars: int,
) -> Iterator[str]:
    """Split a long paragraph by sentences."""
    # Split by sentence boundaries
    # Use individual fixed-width lookbehinds (Python 3.13+ requires fixed-width)
    sentences = re.split(
        r"(?<=[.!?])"  # After sentence-ending punctuation
        r"(?<![A-Z]\.)"  # Not after single capital letter (initials like "J.")
        r"(?<!Dr\.)"
        r"(?<!Mr\.)"
        r"(?<!Ms\.)"
        r"(?<!Mrs\.)"
        r"(?<!Prof\.)"
        r"(?<!Sr\.)"
        r"(?<!Jr\.)"
        r"(?<!vs\.)"
        r"(?<!etc\.)"
        r"(?<!Inc\.)"
        r"(?<!Ltd\.)"
        r"(?<!St\.)"
        r"(?<!Ave\.)"
        r"(?<!Rd\.)"
        r"(?<!Vol\.)"
        r"(?<!No\.)"
        r"(?<!Fig\.)"
        r"\s+",
        text,
    )

    current_chunk: list[str] = []
    current_size = 0
    overlap_text = ""

    for sent in sentences:
        sent_size = len(sent)

        if current_size + sent_size > target_chars and current_chunk:
            chunk_text_out = " ".join(current_chunk)
            if overlap_text:
                chunk_text_out = overlap_text + " " + chunk_text_out
            yield chunk_text_out.strip()

            overlap_text = " ".join(current_chunk)[-overlap_chars:]
            current_chunk = []
            current_size = 0

        current_chunk.append(sent)
        current_size += sent_size

    if current_chunk:
        chunk_text_out = " ".join(current_chunk)
        if overlap_text:
            chunk_text_out = overlap_text + " " + chunk_text_out
        yield chunk_text_out.strip()


def chunk_markdown(
    text: str,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Chunk markdown with section awareness.

    Yields (chunk_text, metadata) tuples where metadata contains:
    - section: The heading this chunk falls under
    """
    if not text.strip():
        return

    # Split by headers (keeping the header with its content)
    sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)

    current_section = ""

    for section in sections:
        if not section.strip():
            continue

        # Extract section header if present
        header_match = re.match(r"^(#{1,6})\s+(.+?)(?:\n|$)", section)
        if header_match:
            current_section = header_match.group(2).strip()

        # Chunk this section
        for chunk in chunk_text(section, target_chars, overlap_chars):
            yield chunk, {"section": current_section}


def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    return len(text) // CHARS_PER_TOKEN
