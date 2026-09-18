"""
Headless-browser crawler.

Given a domain, fetches the homepage, discovers same-domain subpages
that look relevant, and returns raw rendered HTML.

The crawler is designed to be resilient:
- Homepage and subpage failures are isolated.
- HTTP errors are recorded instead of crashing the batch.
- Navigation timeouts are handled.
- JavaScript-rendered pages are supported through Playwright.
- Only same-domain links are accepted.
- Duplicate URLs and fragments are removed.
- A single failed page never stops the remaining crawl.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urljoin, urlparse

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)

logger = logging.getLogger(__name__)


# Relevant pages requested by the assignment plus a few useful
# lead-enrichment pages.
RELEVANT_PATH_KEYWORDS = (
    "about",
    "team",
    "company",
    "contact",
    "pricing",
    "leadership",
    "founder",
    "management",
    "people",
    "careers",
    "press",
)


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class CrawledPage:
    """Successfully crawled page."""

    url: str
    html: str


@dataclass
class CrawlResult:
    """Result of crawling one domain."""

    domain: str
    pages: List[CrawledPage] = field(
        default_factory=list
    )
    errors: List[str] = field(
        default_factory=list
    )

    @property
    def succeeded(self) -> bool:
        """True when at least one page was successfully crawled."""

        return len(self.pages) > 0


def _normalize_domain(domain: str) -> str:
    """Normalize a domain supplied by the user."""

    domain = domain.strip().lower()

    # Remove protocol if supplied.
    if "://" in domain:
        parsed = urlparse(domain)
        domain = parsed.netloc

    # Remove www. so www.example.com and example.com
    # are treated as the same base domain.
    if domain.startswith("www."):
        domain = domain[4:]

    # Remove trailing dot.
    domain = domain.rstrip(".")

    return domain


def _is_same_domain(
    hostname: str | None,
    base_domain: str,
) -> bool:
    """
    Safely check whether a hostname belongs to the target domain.

    This avoids substring mistakes such as accepting:
        evil-example.com
    when the target is:
        example.com
    """

    if not hostname:
        return False

    hostname = hostname.lower().rstrip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    base_domain = _normalize_domain(
        base_domain
    )

    return (
        hostname == base_domain
        or hostname.endswith(
            "." + base_domain
        )
    )


def _is_relevant_link(
    href: str,
    base_domain: str,
) -> bool:
    """
    Return True when href points to the same domain and
    its path contains a relevant keyword.
    """

    try:
        parsed = urlparse(href)
    except ValueError:
        return False

    if parsed.scheme not in (
        "",
        "http",
        "https",
    ):
        return False

    if parsed.netloc and not _is_same_domain(
        parsed.hostname,
        base_domain,
    ):
        return False

    path = parsed.path.lower()

    return any(
        keyword in path
        for keyword in RELEVANT_PATH_KEYWORDS
    )


def _canonicalize_url(
    url: str,
) -> str:
    """
    Normalize URLs so fragments and unnecessary trailing
    differences don't create duplicates.
    """

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    hostname = (
        parsed.hostname.lower()
        if parsed.hostname
        else ""
    )

    if not hostname:
        return url

    # Preserve non-default ports.
    netloc = hostname

    if parsed.port:
        if not (
            (scheme == "https" and parsed.port == 443)
            or (scheme == "http" and parsed.port == 80)
        ):
            netloc = (
                f"{hostname}:{parsed.port}"
            )

    path = parsed.path or "/"

    # Keep query parameters because some websites use them
    # for meaningful pages, but remove fragments.
    return parsed._replace(
        netloc=netloc,
        path=path,
        fragment="",
    ).geturl()


async def _fetch_page(
    context,
    url: str,
    timeout_ms: int,
) -> str:
    """
    Load a single URL and return rendered HTML.

    Raises an exception on failure. The caller is responsible
    for recording the failure and continuing the crawl.
    """

    page = await context.new_page()

    try:
        response = await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )

        if response is None:
            raise RuntimeError(
                f"No response received for {url}"
            )

        if response.status >= 400:
            raise RuntimeError(
                f"HTTP {response.status} for {url}"
            )

        # Give JavaScript a short opportunity to populate
        # dynamic content without waiting for networkidle.
        try:
            await page.wait_for_timeout(500)
        except Exception:
            pass

        html = await page.content()

        if not html.strip():
            raise RuntimeError(
                f"Empty HTML received for {url}"
            )

        return html

    except PlaywrightTimeout as exc:
        raise RuntimeError(
            f"Timeout after {timeout_ms}ms for {url}"
        ) from exc

    finally:
        await page.close()


async def _extract_links(
    context,
    homepage_html: str,
    base_url: str,
    base_domain: str,
    errors: list[str],
) -> list[str]:
    """Extract and rank relevant links from homepage HTML."""

    page = await context.new_page()

    try:
        await page.set_content(
            homepage_html,
            wait_until="domcontentloaded",
        )

        anchors = await page.eval_on_selector_all(
            "a[href]",
            """
            els => els.map(
                e => e.getAttribute("href")
            )
            """,
        )

    except Exception as exc:
        errors.append(
            f"Link extraction failed: {exc}"
        )
        return []

    finally:
        await page.close()

    candidate_links: set[str] = set()

    for href in anchors:
        if not href:
            continue

        href = href.strip()

        if not href:
            continue

        try:
            full_url = urljoin(
                base_url,
                href,
            )

            full_url = _canonicalize_url(
                full_url
            )

        except Exception:
            continue

        if _is_relevant_link(
            full_url,
            base_domain,
        ):
            candidate_links.add(
                full_url
            )

    # Rank links according to how directly they match
    # useful enrichment pages.
    def link_score(url: str) -> int:
        path = urlparse(url).path.lower()

        score = 0

        priority_keywords = (
            "about",
            "team",
            "leadership",
            "founder",
            "company",
            "contact",
            "pricing",
            "people",
            "management",
            "careers",
            "press",
        )

        for index, keyword in enumerate(
            priority_keywords
        ):
            if keyword in path:
                score += (
                    len(priority_keywords)
                    - index
                )

        return score

    return sorted(
        candidate_links,
        key=link_score,
        reverse=True,
    )


async def crawl_domain(
    domain: str,
    max_pages: int = 5,
    timeout_seconds: int = 15,
) -> CrawlResult:
    """
    Crawl a single domain.

    The homepage is always attempted first.
    Up to max_pages - 1 relevant subpages are then attempted.

    If the homepage fails, the crawl stops because there is
    no reliable source for discovering additional links.
    """

    normalized_domain = _normalize_domain(
        domain
    )

    result = CrawlResult(
        domain=normalized_domain
    )

    if not normalized_domain:
        result.errors.append(
            "Empty domain supplied"
        )
        return result

    if max_pages < 1:
        result.errors.append(
            "max_pages must be at least 1"
        )
        return result

    if timeout_seconds <= 0:
        result.errors.append(
            "timeout_seconds must be greater than 0"
        )
        return result

    homepage_url = (
        f"https://{normalized_domain}"
    )

    homepage_url = _canonicalize_url(
        homepage_url
    )

    timeout_ms = (
        timeout_seconds * 1000
    )

    logger.info(
        f"Crawling {normalized_domain}"
    )

    async with async_playwright() as p:

        browser = None

        try:
            browser = await p.chromium.launch(
                headless=True
            )

            context = await browser.new_context(
                user_agent=USER_AGENT,
                java_script_enabled=True,
            )

            # ---------------------------------------------------------
            # 1. Homepage
            # ---------------------------------------------------------

            try:
                homepage_html = await _fetch_page(
                    context,
                    homepage_url,
                    timeout_ms,
                )

                result.pages.append(
                    CrawledPage(
                        url=homepage_url,
                        html=homepage_html,
                    )
                )

                logger.info(
                    f"{normalized_domain}: homepage loaded"
                )

            except Exception as exc:
                error_message = (
                    f"Homepage failed for "
                    f"{normalized_domain}: {exc}"
                )

                logger.warning(
                    error_message
                )

                result.errors.append(
                    error_message
                )

                return result

            # ---------------------------------------------------------
            # 2. Discover relevant links
            # ---------------------------------------------------------

            candidate_links = await _extract_links(
                context=context,
                homepage_html=homepage_html,
                base_url=homepage_url,
                base_domain=normalized_domain,
                errors=result.errors,
            )

            logger.info(
                f"{normalized_domain}: discovered "
                f"{len(candidate_links)} relevant link(s)"
            )

            # ---------------------------------------------------------
            # 3. Fetch relevant pages
            # ---------------------------------------------------------

            pages_remaining = max(
                0,
                max_pages - len(result.pages),
            )

            for link in candidate_links[
                :pages_remaining
            ]:

                # Prevent accidental duplicate crawling.
                already_crawled = any(
                    page.url == link
                    for page in result.pages
                )

                if already_crawled:
                    continue

                try:
                    html = await _fetch_page(
                        context,
                        link,
                        timeout_ms,
                    )

                    result.pages.append(
                        CrawledPage(
                            url=link,
                            html=html,
                        )
                    )

                    logger.info(
                        f"{normalized_domain}: "
                        f"loaded {link}"
                    )

                except Exception as exc:
                    error_message = (
                        f"Subpage failed "
                        f"({link}): {exc}"
                    )

                    logger.warning(
                        error_message
                    )

                    result.errors.append(
                        error_message
                    )

                    # Continue with the next page.
                    continue

            return result

        except Exception as exc:
            # Final safety net: an unexpected browser/context
            # failure should still return a structured result.
            error_message = (
                f"Unexpected crawl error for "
                f"{normalized_domain}: {exc}"
            )

            logger.exception(
                error_message
            )

            result.errors.append(
                error_message
            )

            return result

        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass