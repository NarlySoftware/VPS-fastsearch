"""Tests for vps_fastsearch.chunker — no ML models required."""

from vps_fastsearch.chunker import chunk_markdown, chunk_text, estimate_tokens


def test_chunk_text_multiple_paragraphs():
    """Multiple paragraphs exceeding target should produce multiple chunks."""
    # Build text with many paragraphs (~300 chars each, target is 2000)
    paragraphs = [f"Paragraph {i}. " + "This is filler text. " * 12 for i in range(20)]
    text = "\n\n".join(paragraphs)

    chunks = list(chunk_text(text))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_empty_input():
    """Empty or whitespace-only input yields no chunks."""
    assert list(chunk_text("")) == []
    assert list(chunk_text("   ")) == []
    assert list(chunk_text("\n\n\n")) == []


def test_chunk_text_single_short_paragraph():
    """A short paragraph should return exactly one chunk."""
    text = "This is a short paragraph with just a few words."
    chunks = list(chunk_text(text))
    assert len(chunks) == 1
    assert "short paragraph" in chunks[0]


def test_chunk_markdown_section_metadata():
    """Markdown chunker should extract section headers into metadata."""
    text = "# Introduction\n\nSome intro text.\n\n# Setup\n\nSetup instructions here."
    results = list(chunk_markdown(text))

    assert len(results) >= 1
    # Each result is (chunk_text, metadata)
    for _chunk_text_str, metadata in results:
        assert "section" in metadata
        assert isinstance(metadata["section"], str)

    # Check that we get both sections
    sections = [m["section"] for _, m in results]
    assert "Introduction" in sections
    assert "Setup" in sections


def test_estimate_tokens():
    """Token estimate should be len(text) // 4."""
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == 1  # 5 // 4 = 1
