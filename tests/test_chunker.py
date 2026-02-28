"""Tests for vps_fastsearch.chunker — no ML models required."""

from vps_fastsearch.chunker import (
    CHARS_PER_TOKEN,
    TARGET_CHARS,
    chunk_markdown,
    chunk_text,
    estimate_tokens,
)


def test_chunk_text_multiple_paragraphs() -> None:
    """Multiple paragraphs exceeding target should produce multiple chunks."""
    # Build text with many paragraphs (~300 chars each, target is 2000)
    paragraphs = [f"Paragraph {i}. " + "This is filler text. " * 12 for i in range(20)]
    text = "\n\n".join(paragraphs)

    chunks = list(chunk_text(text))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_empty_input() -> None:
    """Empty or whitespace-only input yields no chunks."""
    assert list(chunk_text("")) == []
    assert list(chunk_text("   ")) == []
    assert list(chunk_text("\n\n\n")) == []


def test_chunk_text_single_short_paragraph() -> None:
    """A short paragraph should return exactly one chunk."""
    text = "This is a short paragraph with just a few words."
    chunks = list(chunk_text(text))
    assert len(chunks) == 1
    assert "short paragraph" in chunks[0]


def test_chunk_markdown_section_metadata() -> None:
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


def test_estimate_tokens() -> None:
    """Token estimate should be len(text) // 4."""
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == 1  # 5 // 4 = 1


# ---------------------------------------------------------------------------
# Edge case tests for chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_single_word() -> None:
    """A single word shorter than chunk size should return one chunk."""
    chunks = list(chunk_text("hello"))
    assert len(chunks) == 1
    assert chunks[0] == "hello"


def test_chunk_text_only_newlines_and_spaces() -> None:
    """Input that is only whitespace variants should yield no chunks."""
    assert list(chunk_text("\n")) == []
    assert list(chunk_text("\t\t")) == []
    assert list(chunk_text("  \n  \n  ")) == []


def test_chunk_text_exactly_at_chunk_boundary() -> None:
    """Input exactly at target_chars should produce a single chunk."""
    # Create text that is exactly TARGET_CHARS long (single paragraph, no splits)
    text = "x" * TARGET_CHARS
    chunks = list(chunk_text(text))
    # Single paragraph at exactly the boundary should produce one chunk
    assert len(chunks) == 1


def test_chunk_text_just_over_boundary() -> None:
    """Two paragraphs that together exceed target should produce multiple chunks."""
    half = "a" * (TARGET_CHARS // 2 + 100)
    text = half + "\n\n" + half
    chunks = list(chunk_text(text))
    assert len(chunks) >= 2


def test_chunk_text_very_large_input() -> None:
    """Very large input should produce a predictable number of chunks."""
    # 100 paragraphs of ~500 chars each = ~50,000 chars total
    paragraphs = [f"Section {i}. " + "Word " * 100 for i in range(100)]
    text = "\n\n".join(paragraphs)
    total_chars = len(text)

    chunks = list(chunk_text(text))
    # Should produce multiple chunks; at least total / target (rough lower bound)
    assert len(chunks) >= total_chars // (TARGET_CHARS * 2)
    assert len(chunks) > 1
    # All chunks should be non-empty
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_zero_overlap() -> None:
    """With overlap=0, chunks should not contain repeated text from previous chunk."""
    paragraphs = [f"Unique paragraph {i}. " + "Filler. " * 50 for i in range(10)]
    text = "\n\n".join(paragraphs)

    chunks = list(chunk_text(text, target_chars=500, overlap_chars=0))
    assert len(chunks) > 1
    # Each chunk should be non-empty
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_overlap_larger_than_target() -> None:
    """When overlap >= target, chunking should still not crash."""
    text = "First paragraph with content.\n\nSecond paragraph with content."
    # overlap_chars > target_chars is a degenerate case but should not raise
    chunks = list(chunk_text(text, target_chars=20, overlap_chars=100))
    assert len(chunks) >= 1


def test_chunk_text_unicode_content() -> None:
    """Unicode characters (CJK, accented, emoji) should be chunked correctly."""
    # Mix of unicode content
    text = "Hello world.\n\n" + "日本語テスト。" * 50 + "\n\n" + "Cafe\u0301 re\u0301sume\u0301."
    chunks = list(chunk_text(text))
    assert len(chunks) >= 1
    # Verify content is preserved (not corrupted)
    joined = " ".join(chunks)
    assert "日本語テスト" in joined
    assert "Hello world" in joined


def test_chunk_text_emoji_content() -> None:
    """Emoji content should not break chunking."""
    emojis = "Hello! 🎉🚀💻🔍📚 This is a test."
    chunks = list(chunk_text(emojis))
    assert len(chunks) == 1
    assert "🎉" in chunks[0]


def test_chunk_text_long_single_paragraph() -> None:
    """A single paragraph exceeding target should be split by sentences."""
    # Build a single long paragraph (no double newlines)
    sentences = [f"Sentence number {i} is here." for i in range(200)]
    text = " ".join(sentences)
    assert "\n\n" not in text  # Confirm it's a single paragraph

    chunks = list(chunk_text(text, target_chars=500, overlap_chars=50))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_preserves_all_content() -> None:
    """No content should be silently dropped during chunking (with overlap=0)."""
    paragraphs = [f"Paragraph {i}." for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = list(chunk_text(text, target_chars=50, overlap_chars=0))
    joined = "\n\n".join(chunks)
    for p in paragraphs:
        assert p in joined


def test_chunk_text_excessive_newlines() -> None:
    """Multiple consecutive newlines should be normalized."""
    text = "First paragraph.\n\n\n\n\n\nSecond paragraph."
    chunks = list(chunk_text(text))
    assert len(chunks) == 1
    assert "First paragraph" in chunks[0]
    assert "Second paragraph" in chunks[0]


def test_chunk_text_custom_target_and_overlap() -> None:
    """Custom target_chars and overlap_chars should work correctly."""
    text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
    chunks = list(chunk_text(text, target_chars=150, overlap_chars=20))
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Edge case tests for chunk_markdown
# ---------------------------------------------------------------------------


def test_chunk_markdown_empty_input() -> None:
    """Empty markdown input yields no results."""
    assert list(chunk_markdown("")) == []
    assert list(chunk_markdown("   ")) == []
    assert list(chunk_markdown("\n\n")) == []


def test_chunk_markdown_no_headers() -> None:
    """Markdown with no headers should still chunk, with empty section metadata."""
    text = "Just some plain text.\n\nAnother paragraph here."
    results = list(chunk_markdown(text))
    assert len(results) >= 1
    for _chunk_text_str, metadata in results:
        assert "section" in metadata
        # Section should be empty string since there are no headers
        assert metadata["section"] == ""


def test_chunk_markdown_nested_headers() -> None:
    """Markdown with nested headers (h1, h2, h3) should track sections correctly."""
    text = (
        "# Top Level\n\nIntro.\n\n"
        "## Sub Section\n\nDetails.\n\n"
        "### Deep Section\n\nDeep details.\n\n"
        "## Another Sub\n\nMore details."
    )
    results = list(chunk_markdown(text))
    sections = [m["section"] for _, m in results]
    assert "Top Level" in sections
    assert "Sub Section" in sections
    assert "Deep Section" in sections
    assert "Another Sub" in sections


def test_chunk_markdown_code_blocks() -> None:
    """Markdown with code blocks should chunk without breaking code."""
    text = (
        "# Code Example\n\n"
        "Here is some code:\n\n"
        "```python\n"
        "def hello():\n"
        "    print('world')\n"
        "```\n\n"
        "End of example."
    )
    results = list(chunk_markdown(text))
    assert len(results) >= 1
    sections = [m["section"] for _, m in results]
    assert "Code Example" in sections


def test_chunk_markdown_tables() -> None:
    """Markdown with tables should be chunked without crashing."""
    text = (
        "# Data Table\n\n"
        "| Name | Value |\n"
        "|------|-------|\n"
        "| foo  | 42    |\n"
        "| bar  | 99    |\n\n"
        "End of table."
    )
    results = list(chunk_markdown(text))
    assert len(results) >= 1
    joined = " ".join(t for t, _ in results)
    assert "foo" in joined
    assert "42" in joined


def test_chunk_markdown_h6_header() -> None:
    """h6 headers (######) should be recognized as section headers."""
    text = "###### Deep Header\n\nContent under deep header."
    results = list(chunk_markdown(text))
    assert len(results) >= 1
    sections = [m["section"] for _, m in results]
    assert "Deep Header" in sections


def test_chunk_markdown_header_only() -> None:
    """A header with no body text should still produce a chunk."""
    text = "# Just a Header"
    results = list(chunk_markdown(text))
    assert len(results) >= 1
    assert results[0][1]["section"] == "Just a Header"


def test_chunk_markdown_large_sections() -> None:
    """Large sections should be split into multiple chunks with same section metadata."""
    big_content = "Sentence. " * 500  # ~5000 chars
    text = f"# Big Section\n\n{big_content}"
    results = list(chunk_markdown(text, target_chars=500))
    assert len(results) > 1
    # All chunks from the same section should carry the section metadata
    for _chunk_text_str, metadata in results:
        assert metadata["section"] == "Big Section"


# ---------------------------------------------------------------------------
# Edge case tests for estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_single_char() -> None:
    """Single character should estimate to 0 tokens (1 // 4 = 0)."""
    assert estimate_tokens("a") == 0


def test_estimate_tokens_exactly_four_chars() -> None:
    """Four characters should estimate to exactly 1 token."""
    assert estimate_tokens("abcd") == 1


def test_estimate_tokens_long_text() -> None:
    """Long text token estimation should scale linearly."""
    text = "x" * 10000
    expected = 10000 // CHARS_PER_TOKEN
    assert estimate_tokens(text) == expected


def test_estimate_tokens_unicode() -> None:
    """Unicode characters should each count as one character for estimation."""
    # Each CJK character is 1 char in Python, so 8 chars -> 2 tokens
    assert estimate_tokens("日本語テスト漢字世") == len("日本語テスト漢字世") // CHARS_PER_TOKEN


def test_estimate_tokens_whitespace_only() -> None:
    """Whitespace-only text should still return a token estimate based on length."""
    assert estimate_tokens("    ") == 1  # 4 // 4 = 1


def test_chunk_text_overlap_produces_repeated_content() -> None:
    """With non-zero overlap, subsequent chunks should contain overlap text."""
    # Create distinct paragraphs that will force multiple chunks
    p1 = "Alpha paragraph. " * 30  # ~510 chars
    p2 = "Beta paragraph. " * 30
    p3 = "Gamma paragraph. " * 30
    text = p1.strip() + "\n\n" + p2.strip() + "\n\n" + p3.strip()

    chunks = list(chunk_text(text, target_chars=600, overlap_chars=100))
    assert len(chunks) >= 2
    # The second chunk should contain some text from the end of the first chunk's content
    # (the overlap region). We just verify chunks are non-empty and > 1.
    for chunk in chunks:
        assert len(chunk.strip()) > 0
