"""
Converts raw page HTML into clean, LLM-friendly markdown.

Strips scripts, styles, SVGs, navigation/footer boilerplate, and
other non-semantic content before the extraction stage.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md


# Tags that carry no useful semantic content for our extraction task.
STRIP_TAGS = [
    "script",
    "style",
    "svg",
    "noscript",
    "iframe",
    "nav",
    "footer",
    "form",
]


# Rough character-to-token ratio for local truncation.
CHARS_PER_TOKEN_ESTIMATE = 4


def _clean_markdown_artifacts(text: str) -> str:
    """Remove common Markdown conversion artifacts and encoding issues."""

    # Fix common UTF-8 / Windows-1252 mojibake.
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€”": "-",
        "â€“": "-",
        "Â®": "®",
        "Â©": "©",
        "Â·": "·",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove escaped Markdown characters created by some websites.
    text = re.sub(r"\\([\\`*{}\[\]()#+.!_>|~-])", r"\1", text)

    # Remove leftover backslashes before ordinary letters.
    text = re.sub(r"\\(?=[A-Za-z])", "", text)

    # Convert escaped quotes.
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")

    return text


def html_to_clean_markdown(
    html: str,
    max_tokens: int = 3000,
) -> str:
    """
    Strip boilerplate from raw HTML and convert the remainder to
    clean Markdown, truncated to approximately max_tokens.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove tags that do not contribute useful company information.
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements hidden from users.
    for tag in soup.find_all(
        attrs={"aria-hidden": "true"}
    ):
        tag.decompose()

    body = soup.body or soup

    markdown_text = md(
        str(body),
        heading_style="ATX",
    )

    markdown_text = _clean_markdown_artifacts(
        markdown_text
    )

    # Normalize line endings.
    markdown_text = markdown_text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Remove trailing whitespace.
    lines = [
        line.rstrip()
        for line in markdown_text.splitlines()
    ]

    # Collapse excessive blank lines.
    collapsed = []
    blank_streak = 0

    for line in lines:
        if line.strip() == "":
            blank_streak += 1

            if blank_streak > 1:
                continue
        else:
            blank_streak = 0

        collapsed.append(line)

    cleaned = "\n".join(
        collapsed
    ).strip()

    # Remove empty Markdown link/image remnants.
    cleaned = re.sub(
        r"!\[\]\([^)]+\)",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\[\]\([^)]+\)",
        "",
        cleaned,
    )

    # Normalize repeated spaces without destroying line structure.
    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    cleaned = cleaned.strip()

    # Local truncation to control context size.
    max_chars = (
        max_tokens
        * CHARS_PER_TOKEN_ESTIMATE
    )

    if len(cleaned) > max_chars:
        cleaned = (
            cleaned[:max_chars]
            + "\n\n[...truncated...]"
        )

    return cleaned


def combine_pages(
    pages: list[tuple[str, str]],
    max_tokens_total: int = 8000,
) -> str:
    """
    Combine (url, html) pairs into one Markdown blob for the LLM.

    Each page receives a portion of the overall token budget so that
    one very large page cannot dominate the model context.
    """

    if not pages:
        return ""

    per_page_budget = max(
        500,
        max_tokens_total // len(pages),
    )

    sections = []

    for url, html in pages:
        cleaned = html_to_clean_markdown(
            html,
            max_tokens=per_page_budget,
        )

        sections.append(
            f"## Source: {url}\n\n{cleaned}"
        )

    return "\n\n---\n\n".join(
        sections
    )


def extract_emails(
    text: str,
) -> list[str]:
    """Extract unique email addresses from text."""

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    emails = re.findall(
        pattern,
        text,
    )

    return sorted(
        set(emails),
        key=str.lower,
    )