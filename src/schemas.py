"""
Pydantic models describing the structured data we want the LLM to extract
for each company domain, plus the crawl metadata we attach ourselves.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class TeamMember(BaseModel):
    name: str = Field(description="Full name of the person")
    role: Optional[str] = Field(
        default=None, description="Their title/role, e.g. 'CEO & Co-founder'"
    )
    linkedin_url: Optional[str] = Field(
        default=None, description="LinkedIn profile URL if discoverable on the page"
    )


class CompanyIntelligence(BaseModel):
    """The core structured payload the LLM must produce for a single company."""

    company_overview: str = Field(
        description="A concise 2-sentence summary of what the company does"
    )
    target_audience: str = Field(
        description="Who the product is built for, e.g. 'Developers building backend APIs'"
    )
    contact_points: List[str] = Field(
        default_factory=list,
        description="Generic public emails found on the site (contact@, sales@, support@, etc.)",
    )
    key_leadership: List[TeamMember] = Field(
        default_factory=list,
        description="Names, roles, and LinkedIn URLs of leadership/team members mentioned",
    )
    data_confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0-1.0 score estimating completeness/quality of the extracted data",
    )


class DomainResult(BaseModel):
    """Top-level record written to output.json for each domain — includes
    crawl metadata and error state so the pipeline never silently loses a domain."""

    domain: str
    status: str = Field(description="'success' or 'failed'")
    pages_crawled: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    data: Optional[CompanyIntelligence] = None
    tokens_used: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
