"""
Entry point for the lead-enrichment pipeline.

Usage:
    python -m src.main
    python -m src.main --domains postman.com supabase.com vapi.ai
    python -m src.main --domains postman.com --mock

Outputs:
    output/output.json
    output/output.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.crawler import crawl_domain
from src.cleaner import (
    combine_pages,
    extract_emails,
    html_to_clean_markdown,
)
from src.extractor import extract_company_intelligence
from src.schemas import DomainResult


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DEFAULT_DOMAINS = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]


OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
)


OUTPUT_JSON_PATH = (
    OUTPUT_DIR
    / "output.json"
)


OUTPUT_CSV_PATH = (
    OUTPUT_DIR
    / "output.csv"
)


# ---------------------------------------------------------------------
# PROCESS ONE DOMAIN
# ---------------------------------------------------------------------

async def process_domain(
    domain: str,
    max_pages: int,
    timeout_s: int,
    mock: bool = False,
) -> DomainResult:
    """Run the complete enrichment pipeline for one domain."""

    logger.info(
        f"Starting: {domain}"
    )

    # ---------------------------------------------------------------
    # 1. Crawl
    # ---------------------------------------------------------------

    try:
        crawl_result = await crawl_domain(
            domain,
            max_pages=max_pages,
            timeout_seconds=timeout_s,
        )

    except Exception as e:
        logger.error(
            f"Crawl crashed for {domain}: {e}"
        )

        return DomainResult(
            domain=domain,
            status="failed",
            error=f"Crawl error: {e}",
        )

    if not crawl_result.succeeded:

        error_msg = (
            "; ".join(
                crawl_result.errors
            )
            or "No pages could be crawled"
        )

        logger.warning(
            f"No pages crawled for "
            f"{domain}: {error_msg}"
        )

        return DomainResult(
            domain=domain,
            status="failed",
            error=error_msg,
        )

    pages_crawled = [
        page.url
        for page in crawl_result.pages
    ]

    logger.info(
        f"{domain}: crawled "
        f"{len(pages_crawled)} page(s)"
    )

    # ---------------------------------------------------------------
    # 2. Clean HTML and extract emails
    # ---------------------------------------------------------------

    try:
        cleaned_pages = []

        for page in crawl_result.pages:

            page_markdown = (
                html_to_clean_markdown(
                    page.html
                )
            )

            cleaned_pages.append(
                (
                    page.url,
                    page_markdown,
                )
            )

        # Extract emails BEFORE combined Markdown is truncated.
        # This prevents emails near the end of a page from being lost.
        emails = sorted(
            {
                email
                for _, page_markdown
                in cleaned_pages
                for email in extract_emails(
                    page_markdown
                )
            },
            key=str.lower,
        )

        cleaned_markdown = combine_pages(
            [
                (
                    url,
                    markdown,
                )
                for url, markdown
                in cleaned_pages
            ]
        )

        logger.info(
            f"{domain}: found "
            f"{len(emails)} public email(s)"
        )

    except Exception as e:

        logger.error(
            f"Cleaning failed for "
            f"{domain}: {e}"
        )

        return DomainResult(
            domain=domain,
            status="failed",
            pages_crawled=pages_crawled,
            error=f"Cleaning error: {e}",
        )

    # ---------------------------------------------------------------
    # 3. Structured extraction
    # ---------------------------------------------------------------

    try:

        data, tokens_used, cost = (
            extract_company_intelligence(
                domain,
                cleaned_markdown,
                mock=mock,
            )
        )

        # Deterministic email extraction is more reliable for
        # exact public email addresses, so use it when available.
        if emails:
            data.contact_points = emails

    except Exception as e:

        logger.error(
            f"LLM extraction failed for "
            f"{domain}: {e}"
        )

        return DomainResult(
            domain=domain,
            status="failed",
            pages_crawled=pages_crawled,
            error=f"Extraction error: {e}",
        )

    # ---------------------------------------------------------------
    # 4. Log result
    # ---------------------------------------------------------------

    logger.info(
        f"{domain}: done "
        f"(tokens={tokens_used}, "
        f"est_cost=${cost:.4f})"
    )

    return DomainResult(
        domain=domain,
        status="success",
        pages_crawled=pages_crawled,
        data=data,
        tokens_used=tokens_used,
        estimated_cost_usd=round(
            cost,
            6,
        ),
    )


# ---------------------------------------------------------------------
# PROCESS ALL DOMAINS
# ---------------------------------------------------------------------

async def run_pipeline(
    domains: list[str],
    max_pages: int,
    timeout_s: int,
    mock: bool = False,
) -> list[DomainResult]:
    """Process all domains sequentially."""

    results: list[DomainResult] = []

    for domain in domains:

        result = await process_domain(
            domain,
            max_pages,
            timeout_s,
            mock=mock,
        )

        results.append(
            result
        )

    return results


# ---------------------------------------------------------------------
# JSON OUTPUT
# ---------------------------------------------------------------------

def write_json_output(
    results: list[DomainResult],
) -> None:
    """Write structured results to output.json."""

    output_data = [
        result.model_dump()
        for result in results
    ]

    with open(
        OUTPUT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        f"JSON output written to "
        f"{OUTPUT_JSON_PATH}"
    )


# ---------------------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------------------

def _leadership_to_csv(
    result: DomainResult,
) -> str:
    """
    Convert leadership records into a compact CSV-friendly string.

    Example:
        Abhinav Asthana (CEO); Jane Doe (Founder)
    """

    if not result.data:
        return ""

    members = result.data.key_leadership

    if not members:
        return ""

    leadership_items: list[str] = []

    for member in members:

        name = member.name.strip()

        role = (
            member.role.strip()
            if member.role
            else ""
        )

        linkedin = (
            member.linkedin_url.strip()
            if member.linkedin_url
            else ""
        )

        if role and linkedin:

            leadership_items.append(
                f"{name} ({role}) - {linkedin}"
            )

        elif role:

            leadership_items.append(
                f"{name} ({role})"
            )

        elif linkedin:

            leadership_items.append(
                f"{name} - {linkedin}"
            )

        else:

            leadership_items.append(
                name
            )

    return "; ".join(
        leadership_items
    )


def _pages_to_csv(
    result: DomainResult,
) -> str:
    """Convert crawled page URLs into one CSV cell."""

    return "; ".join(
        result.pages_crawled
    )


def _error_to_csv(
    result: DomainResult,
) -> str:
    """Return error text for CSV output."""

    if not result.error:
        return ""

    return result.error


def build_csv_rows(
    results: list[DomainResult],
) -> list[dict[str, object]]:
    """Convert structured results into flat CSV rows."""

    rows: list[dict[str, object]] = []

    for result in results:

        company_overview = ""
        target_audience = ""
        contact_points = ""
        leadership = ""
        confidence = ""

        if result.data:

            company_overview = (
                result.data.company_overview
            )

            target_audience = (
                result.data.target_audience
            )

            contact_points = (
                "; ".join(
                    result.data.contact_points
                )
            )

            leadership = (
                _leadership_to_csv(
                    result
                )
            )

            confidence = (
                result.data
                .data_confidence_score
            )

        rows.append(
            {
                "domain": result.domain,
                "status": result.status,
                "pages_crawled": len(
                    result.pages_crawled
                ),
                "page_urls": _pages_to_csv(
                    result
                ),
                "company_overview": (
                    company_overview
                ),
                "target_audience": (
                    target_audience
                ),
                "contact_points": (
                    contact_points
                ),
                "key_leadership": (
                    leadership
                ),
                "data_confidence_score": (
                    confidence
                ),
                "tokens_used": (
                    result.tokens_used or 0
                ),
                "estimated_cost_usd": (
                    result.estimated_cost_usd
                    or 0.0
                ),
                "error": _error_to_csv(
                    result
                ),
            }
        )

    return rows


def write_csv_output(
    results: list[DomainResult],
) -> None:
    """Write flat results to output.csv."""

    rows = build_csv_rows(
        results
    )

    fieldnames = [
        "domain",
        "status",
        "pages_crawled",
        "page_urls",
        "company_overview",
        "target_audience",
        "contact_points",
        "key_leadership",
        "data_confidence_score",
        "tokens_used",
        "estimated_cost_usd",
        "error",
    ]

    with open(
        OUTPUT_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    logger.info(
        f"CSV output written to "
        f"{OUTPUT_CSV_PATH}"
    )


# ---------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------

def main():
    """CLI entry point."""

    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Autonomous lead enrichment agent"
        )
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help=(
            "Company domains to enrich"
        ),
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Run without calling the "
            "Anthropic API"
        ),
    )

    args = parser.parse_args()

    max_pages = int(
        os.environ.get(
            "MAX_PAGES_PER_DOMAIN",
            5,
        )
    )

    timeout_s = int(
        os.environ.get(
            "PAGE_TIMEOUT_SECONDS",
            15,
        )
    )

    # ---------------------------------------------------------------
    # API key validation
    # ---------------------------------------------------------------

    if (
        not args.mock
        and not os.environ.get(
            "ANTHROPIC_API_KEY"
        )
    ):

        raise SystemExit(
            "ANTHROPIC_API_KEY not set. "
            "Copy .env.example to .env "
            "and add your key."
        )

    # ---------------------------------------------------------------
    # Run pipeline
    # ---------------------------------------------------------------

    start = time.time()

    results = asyncio.run(
        run_pipeline(
            args.domains,
            max_pages,
            timeout_s,
            mock=args.mock,
        )
    )

    elapsed = (
        time.time()
        - start
    )

    # ---------------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Write JSON + CSV
    # ---------------------------------------------------------------

    write_json_output(
        results
    )

    write_csv_output(
        results
    )

    # ---------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------

    succeeded = sum(
        1
        for result in results
        if result.status == "success"
    )

    failed = (
        len(results)
        - succeeded
    )

    total_tokens = sum(
        result.tokens_used or 0
        for result in results
    )

    total_cost = sum(
        result.estimated_cost_usd or 0
        for result in results
    )

    logger.info(
        f"Done in {elapsed:.1f}s | "
        f"{succeeded}/{len(results)} succeeded | "
        f"{failed} failed | "
        f"tokens={total_tokens} | "
        f"est. cost=${total_cost:.4f}"
    )


if __name__ == "__main__":
    main()