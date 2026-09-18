"""
Extract structured company intelligence from cleaned website content.

Normal mode:
    Uses Claude with structured tool calling.

Mock mode:
    Uses deterministic local extraction so the pipeline can be tested
    without Anthropic API credits.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from anthropic import Anthropic

from src.schemas import CompanyIntelligence, TeamMember


# Claude Sonnet pricing used for estimated cost tracking.
PRICE_PER_MTOK_INPUT = 3.00
PRICE_PER_MTOK_OUTPUT = 15.00


EXTRACTION_TOOL = {
    "name": "record_company_intelligence",
    "description": (
        "Record structured intelligence extracted about a company "
        "from its website content."
    ),
    "input_schema": CompanyIntelligence.model_json_schema(),
}


SYSTEM_PROMPT = """You are a precise B2B research analyst.

You will be given cleaned markdown content scraped from a company's
public website.

Extract only information that is explicitly present or strongly
supported by the source content. Do not invent names, emails,
company facts, or LinkedIn URLs.

For missing fields, return an empty string or empty list.

The company overview must contain exactly two concise sentences
describing what the company does and the main product or service
it provides.

The target audience should describe the customers, users, developers,
teams, businesses, or organizations the company serves.

For contact_points, include publicly listed generic or business
contact emails found in the source.

For key_leadership, include people whose names and leadership roles
are explicitly identified in the source.

Include a LinkedIn URL only when it is explicitly available in
the supplied source content.

Set data_confidence_score between 0.0 and 1.0 based on the
completeness and reliability of the extracted information."""


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate Claude API cost from token usage."""

    return (
        (input_tokens / 1_000_000) * PRICE_PER_MTOK_INPUT
        + (output_tokens / 1_000_000) * PRICE_PER_MTOK_OUTPUT
    )


# ---------------------------------------------------------------------
# TEXT CLEANING HELPERS
# ---------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable text."""

    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Fix common markdown conversion artifacts.
    text = re.sub(
        r"\bpartof\b",
        "part of",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _remove_markdown_links(text: str) -> str:
    """Convert markdown links to their visible text."""

    return re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )


def _clean_candidate_text(text: str) -> str:
    """Clean common markdown and navigation artifacts."""

    text = _remove_markdown_links(text)

    text = re.sub(
        r"^#+\s*",
        "",
        text,
    )

    text = _normalize_text(text)

    return text.strip()


# ---------------------------------------------------------------------
# EMAIL EXTRACTION
# ---------------------------------------------------------------------

def _extract_mock_emails(
    cleaned_markdown: str,
) -> list[str]:
    """Extract unique public email addresses from website text."""

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    emails = re.findall(
        pattern,
        cleaned_markdown,
    )

    return sorted(
        set(emails),
        key=str.lower,
    )


# ---------------------------------------------------------------------
# COMPANY OVERVIEW
# ---------------------------------------------------------------------

def _is_bad_overview_line(
    line: str,
) -> bool:
    """Return True when a line is unlikely to describe the company."""

    lower = line.lower()

    blocked_phrases = [
        "source:",
        "skip to",
        "learn more",
        "register now",
        "sign up",
        "sign in",
        "log in",
        "cookie",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "accept cookies",
        "menu",
        "navigation",
        "search",
        "subscribe",
        "contact us",
        "contact sales",
        "book a demo",
        "get started",
        "request a demo",
        "vapicon is back",
        "vapiCon is back".lower(),
    ]

    return any(
        phrase in lower
        for phrase in blocked_phrases
    )


def _looks_like_company_description(
    line: str,
) -> bool:
    """Check whether a line resembles a company/product description."""

    lower = line.lower()

    description_patterns = [
        " is a ",
        " is an ",
        " is the ",
        "provides ",
        "provides a ",
        "provides an ",
        "helps ",
        "helping ",
        "platform for ",
        "platform that ",
        "platform to ",
        "software for ",
        "software that ",
        "tools for ",
        "tools that ",
        "solution for ",
        "solutions for ",
        "enables ",
        "allows ",
        "designed for ",
        "built for ",
        "build ",
        "builds ",
        "create ",
        "creates ",
        "api platform",
        "cloud platform",
        "database platform",
        "voice ai",
        "artificial intelligence",
    ]

    return any(
        pattern in lower
        for pattern in description_patterns
    )


def _extract_mock_overview(
    cleaned_markdown: str,
    domain: str,
) -> str:
    """
    Extract a concise company overview from descriptive website text.

    This intentionally uses deterministic rules in mock mode.
    """

    raw_lines = [
        line.strip()
        for line in cleaned_markdown.splitlines()
        if line.strip()
    ]

    candidates: list[str] = []

    for raw_line in raw_lines:
        line = _clean_candidate_text(
            raw_line
        )

        if not line:
            continue

        if _is_bad_overview_line(line):
            continue

        if line.startswith("!["):
            continue

        if (
            line.startswith("http://")
            or line.startswith("https://")
        ):
            continue

        # Ignore short headings/navigation.
        if len(line) < 70:
            continue

        # Extremely long lines are usually concatenated navigation.
        if len(line) > 700:
            continue

        if _looks_like_company_description(line):
            candidates.append(line)

    # Remove duplicate candidates.
    unique_candidates: list[str] = []

    for candidate in candidates:
        if not any(
            candidate.lower() == existing.lower()
            for existing in unique_candidates
        ):
            unique_candidates.append(candidate)

    # Fallback when no explicit description pattern was found.
    if not unique_candidates:
        for raw_line in raw_lines:
            line = _clean_candidate_text(
                raw_line
            )

            if (
                len(line) >= 100
                and len(line) <= 500
                and not _is_bad_overview_line(line)
            ):
                unique_candidates.append(line)

                if len(unique_candidates) == 2:
                    break

    if not unique_candidates:
        return (
            f"Public website information was collected for "
            f"{domain}."
        )

    selected = unique_candidates[:2]

    overview = " ".join(selected)

    # Keep at most two sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        overview,
    )

    if len(sentences) > 2:
        overview = " ".join(
            sentences[:2]
        )

    return overview.strip()


# ---------------------------------------------------------------------
# TARGET AUDIENCE
# ---------------------------------------------------------------------

def _extract_mock_target_audience(
    cleaned_markdown: str,
) -> str:
    """Extract likely target-audience references."""

    text = _normalize_text(
        cleaned_markdown
    )

    audience_patterns = [
        (
            r"\bsoftware developers?\b",
            "software developers",
        ),
        (
            r"\bdevelopers?\b",
            "developers",
        ),
        (
            r"\bengineering teams?\b",
            "engineering teams",
        ),
        (
            r"\bsoftware teams?\b",
            "software teams",
        ),
        (
            r"\bproduct teams?\b",
            "product teams",
        ),
        (
            r"\bstartups?\b",
            "startups",
        ),
        (
            r"\benterprises?\b",
            "enterprise organizations",
        ),
        (
            r"\bbusinesses?\b",
            "businesses",
        ),
        (
            r"\bcompanies\b",
            "companies",
        ),
    ]

    found: list[str] = []

    for pattern, label in audience_patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            if label not in found:
                found.append(label)

    # Remove generic "developers" when "software developers"
    # is already present.
    if "software developers" in found:
        found = [
            item
            for item in found
            if item != "developers"
        ]

    if not found:
        return ""

    if len(found) == 1:
        return (
            f"The website references {found[0]} "
            "as part of its target audience."
        )

    if len(found) == 2:
        audience_text = (
            f"{found[0]} and {found[1]}"
        )
    else:
        audience_text = (
            ", ".join(found[:-1])
            + f", and {found[-1]}"
        )

    return (
        f"The website references {audience_text} "
        "as part of its target audience."
    )


# ---------------------------------------------------------------------
# LEADERSHIP EXTRACTION
# ---------------------------------------------------------------------

def _clean_person_name(
    name: str,
) -> str:
    """Clean a captured person's name."""

    name = _normalize_text(
        name
    )

    # Remove common testimonial/customer company suffixes.
    name = re.sub(
        r"\s+(?:Infinitus|Makerpad|LogDNA|Vercel|GitHub|Fly|Pipedream)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = name.strip(
        " \t\n\r,.;:-—–"
    )

    return name


def _clean_role(
    role: str,
) -> str:
    """Normalize a leadership role."""

    role = _normalize_text(
        role
    )

    if role.lower() == "co-founder":
        return "Co-founder"

    return role.strip(
        " \t\n\r,.;:-—–"
    )


def _add_leadership_member(
    members: list[TeamMember],
    name: str,
    role: str,
) -> None:
    """Add a valid leadership member if not already present."""

    name = _clean_person_name(
        name
    )

    role = _clean_role(
        role
    )

    if not name or not role:
        return

    # A valid person name normally contains at least two words.
    if len(name.split()) < 2:
        return

    # Reject fragments accidentally captured from phrases such as
    # "CEO and co-founder".
    if name.lower() in {
        "and co",
        "and co founder",
        "and co-founder",
        "co founder",
        "co-founder",
    }:
        return

    if len(name) > 60:
        return

    # Reject obvious navigation/UI phrases.
    blocked_names = {
        "learn more",
        "contact sales",
        "contact us",
        "get started",
        "sign up",
        "sign in",
        "log in",
        "our team",
        "the team",
    }

    if name.lower() in blocked_names:
        return

    # Reject names containing obvious sentence fragments.
    if any(
        phrase in name.lower()
        for phrase in [
            " and ",
            " the ",
            " for ",
            " with ",
        ]
    ):
        return

    duplicate = any(
        existing.name.lower() == name.lower()
        for existing in members
    )

    if duplicate:
        return

    members.append(
        TeamMember(
            name=name,
            role=role,
        )
    )


def _extract_mock_leadership(
    cleaned_markdown: str,
) -> list[TeamMember]:
    """
    Extract likely founders and executives.

    The mock extractor uses conservative patterns and does not invent
    leadership information.
    """

    members: list[TeamMember] = []

    role_terms = (
        r"CEO|CTO|CFO|COO|"
        r"Chief Executive Officer|"
        r"Chief Technology Officer|"
        r"Chief Financial Officer|"
        r"Chief Operating Officer|"
        r"President|"
        r"Founder|"
        r"Co-founder|"
        r"Co-Founder"
    )

    # ---------------------------------------------------------------
    # Pattern 1
    #
    # Examples:
    # "Abhinav Asthana, Postman's CEO and co-founder"
    # "Jane Doe, CEO"
    # "Jane Doe, Founder"
    # ---------------------------------------------------------------

    pattern_1 = (
        r"\b("
        r"[A-Z][A-Za-z'-]+"
        r"(?:\s+[A-Z][A-Za-z'-]+){1,3}"
        r")"
        r",\s+"
        r"(?:[A-Za-z0-9'-]+(?:'s)?\s+)?"
        r"("
        + role_terms
        + r")"
        r"(?:\s+and\s+(?:co-founder|Co-Founder|founder))?"
    )

    for match in re.finditer(
        pattern_1,
        cleaned_markdown,
        flags=re.IGNORECASE,
    ):
        name = match.group(1)
        role = match.group(2)

        full_match = match.group(0).lower()

        if "co-founder" in full_match:
            role = f"{role} and Co-founder"

        _add_leadership_member(
            members,
            name,
            role,
        )

    # ---------------------------------------------------------------
    # Pattern 2
    #
    # Examples:
    # "Jane Doe — CEO"
    # "Jane Doe - Founder"
    # "Jane Doe: CTO"
    # ---------------------------------------------------------------

    pattern_2 = (
        r"\b("
        r"[A-Z][A-Za-z'-]+"
        r"(?:\s+[A-Z][A-Za-z'-]+){1,3}"
        r")"
        r"\s*(?:—|–|-|:)\s*"
        r"("
        + role_terms
        + r")"
    )

    for match in re.finditer(
        pattern_2,
        cleaned_markdown,
        flags=re.IGNORECASE,
    ):
        _add_leadership_member(
            members,
            match.group(1),
            match.group(2),
        )

    # ---------------------------------------------------------------
    # Pattern 3
    #
    # Examples:
    # "CEO Jane Doe"
    # "Founder Jane Doe"
    # ---------------------------------------------------------------

    pattern_3 = (
        r"\b("
        + role_terms
        + r")"
        r"\s+"
        r"("
        r"[A-Z][A-Za-z'-]+"
        r"(?:\s+[A-Z][A-Za-z'-]+){1,3}"
        r")"
    )

    for match in re.finditer(
        pattern_3,
        cleaned_markdown,
        flags=re.IGNORECASE,
    ):
        _add_leadership_member(
            members,
            match.group(2),
            match.group(1),
        )

    return members


# ---------------------------------------------------------------------
# MOCK STRUCTURED EXTRACTION
# ---------------------------------------------------------------------

def _mock_extract_company_intelligence(
    domain: str,
    cleaned_markdown: str,
) -> Tuple[
    CompanyIntelligence,
    int,
    float,
]:
    """Create structured output locally for offline testing."""

    if not cleaned_markdown.strip():
        raise ValueError(
            f"No content scraped for {domain}; nothing to extract."
        )

    emails = _extract_mock_emails(
        cleaned_markdown
    )

    leadership = _extract_mock_leadership(
        cleaned_markdown
    )

    company_overview = _extract_mock_overview(
        cleaned_markdown,
        domain,
    )

    target_audience = _extract_mock_target_audience(
        cleaned_markdown
    )

    # Conservative confidence calculation.
    confidence = 0.50

    if company_overview:
        confidence += 0.15

    if target_audience:
        confidence += 0.10

    if emails:
        confidence += 0.10

    if leadership:
        confidence += 0.10

    confidence = round(
        min(confidence, 1.0),
        2,
    )

    data = CompanyIntelligence(
        company_overview=company_overview,
        target_audience=target_audience,
        contact_points=emails,
        key_leadership=leadership,
        data_confidence_score=confidence,
    )

    return (
        data,
        0,
        0.0,
    )


# ---------------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------------------

def extract_company_intelligence(
    domain: str,
    cleaned_markdown: str,
    client: Optional[Anthropic] = None,
    mock: bool = False,
) -> Tuple[
    CompanyIntelligence,
    int,
    float,
]:
    """
    Extract company intelligence.

    Normal mode:
        Calls Claude using structured tool calling.

    Mock mode:
        Performs deterministic local extraction without an API call.
    """

    if mock:
        return _mock_extract_company_intelligence(
            domain,
            cleaned_markdown,
        )

    if not cleaned_markdown.strip():
        raise ValueError(
            f"No content scraped for {domain}; nothing to extract."
        )

    client = client or Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    model = os.environ.get(
        "ANTHROPIC_MODEL",
        "claude-sonnet-4-5",
    )

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        tools=[
            EXTRACTION_TOOL
        ],
        tool_choice={
            "type": "tool",
            "name": "record_company_intelligence",
        },
        messages=[
            {
                "role": "user",
                "content": (
                    f"Company domain: {domain}\n\n"
                    f"Scraped website content:\n\n"
                    f"{cleaned_markdown}"
                ),
            }
        ],
    )

    tool_use_block = next(
        (
            block
            for block in response.content
            if block.type == "tool_use"
        ),
        None,
    )

    if tool_use_block is None:
        raise RuntimeError(
            f"Model did not return a tool_use block for {domain}"
        )

    data = CompanyIntelligence.model_validate(
        tool_use_block.input
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    total_tokens = (
        input_tokens
        + output_tokens
    )

    cost = _estimate_cost(
        input_tokens,
        output_tokens,
    )

    return (
        data,
        total_tokens,
        cost,
    )