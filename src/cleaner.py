"""
Converts raw page HTML into clean, LLM-friendly markdown.

Strips scripts, styles, SVGs, nav/footer boilerplate, and other noise
before conversion — this is what keeps token usage (and cost/latency) down.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Tags that carry no useful semantic content for our extraction task
STRIP_TAGS = ["script", "style", "svg", "noscript", "iframe", "nav", "footer", "form"]

# Rough char-to-token ratio for a cheap local truncation heuristic (~4 chars/token)
CHARS_PER_TOKEN_ESTIMATE = 4


def html_to_clean_markdown(html: str, max_tokens: int = 3000) -> str:
    """Strip boilerplate from raw HTML and convert the remainder to markdown,
    truncated to roughly max_tokens tokens."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Drop elements Playwright/websites commonly use purely for layout/tracking
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()

    body = soup.body or soup
    markdown_text = md(str(body), heading_style="ATX")

    # Collapse excessive blank lines left behind by stripped elements
    lines = [line.rstrip() for line in markdown_text.splitlines()]
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
    cleaned = "\n".join(collapsed).strip()

    max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n\n[...truncated...]"

    return cleaned


def combine_pages(pages: list[tuple[str, str]], max_tokens_total: int = 8000) -> str:
    """Combine (url, html) pairs into one markdown blob for the LLM, with a
    per-page token budget so no single page dominates the context."""
    if not pages:
        return ""

    per_page_budget = max(500, max_tokens_total // len(pages))
    sections = []
    for url, html in pages:
        cleaned = html_to_clean_markdown(html, max_tokens=per_page_budget)
        sections.append(f"## Source: {url}\n\n{cleaned}")

    return "\n\n---\n\n".join(sections)
def extract_emails(text: str) -> list[str]:
    """Extract unique email addresses from text."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(pattern, text)

    return sorted(set(emails), key=str.lower)