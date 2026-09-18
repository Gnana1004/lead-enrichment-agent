# Autonomous Lead Enrichment Agent

An autonomous AI-powered lead enrichment agent that crawls a company's public web presence, extracts relevant information, and produces structured company intelligence.

The system accepts company domains and automatically:

1. Crawls the company homepage.
2. Discovers relevant subpages such as About, Company, Contact, Pricing, Careers, Leadership, and Team pages.
3. Handles JavaScript-rendered pages using Playwright.
4. Cleans HTML content into Markdown/text before processing.
5. Extracts public contact emails.
6. Uses an LLM to generate structured company intelligence.
7. Identifies target audience and key leadership/team members.
8. Assigns a confidence score to the extracted information.
9. Handles failures such as 404 pages, timeouts, blocked pages, and missing content without stopping the complete batch.
10. Tracks token usage and estimated LLM cost.

---

## Assignment

This project was developed as a practical implementation of an autonomous lead enrichment agent.

### Test Domains

The agent was tested against:

- `postman.com`
- `supabase.com`
- `vapi.ai`

### Required Output

For each company, the agent produces:

- Company overview
- Target audience / ICP
- Public contact emails
- Key leadership/team members
- Roles
- LinkedIn URLs when publicly discoverable
- Data confidence score
- Crawled pages
- Processing status
- Error information
- Token usage
- Estimated cost

---

# Architecture

```text
                  Company Domains
                        |
                        v
              +--------------------+
              |    Main Pipeline   |
              +--------------------+
                        |
                        v
              +--------------------+
              |  Playwright Crawler|
              +--------------------+
                        |
             +----------+----------+
             |                     |
             v                     v
       Homepage              Relevant Links
                              |
                              v
                    About / Company /
                    Contact / Pricing /
                    Team / Careers /
                    Leadership / People
                              |
                              v
              +--------------------+
              |   HTML Cleaner     |
              +--------------------+
                        |
                        v
              HTML -> Clean Markdown
                        |
                        v
              +--------------------+
              | Structured LLM     |
              | Extraction         |
              +--------------------+
                        |
                        v
              +--------------------+
              | Pydantic Schema    |
              +--------------------+
                        |
                        v
                output.json
                output.csv
lead-enrichment-agent/
│
├── src/
│   ├── main.py
│   ├── crawler.py
│   ├── cleaner.py
│   ├── extractor.py
│   └── schemas.py
│
├── output/
│   ├── output.json
│   └── output.csv
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

| Component             | Technology       |
| --------------------- | ---------------- |
| Programming Language  | Python 3.12      |
| Web Crawling          | Playwright       |
| HTML Parsing          | BeautifulSoup    |
| HTML to Markdown      | Markdownify      |
| LLM                   | Anthropic Claude |
| Structured Validation | Pydantic         |
| Environment Variables | python-dotenv    |
| Output Formats        | JSON, CSV        |



Requirements
Python 3.12
pip
Internet connection
Anthropic API key for live LLM extraction
Playwright Chromium browser

Python 3.12 is recommended because some dependencies may not build correctly on newer Python versions.