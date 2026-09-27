
# Credit Agent: Automated Financial Analysis Platform

An end-to-end credit underwriting tool that ingests SEC 10-K filings and produces
structured financial data, credit metrics, and a full underwriting memo in under 90 seconds.

Built for commercial credit analysts to eliminate manual spreading and accelerate
the underwriting process.

---

## What it does

1. **Ingests** 10-K filings (HTML or PDF, up to 30MB)
2. **Extracts** structured financials across 3 periods (income statement, balance sheet, cash flow) using GPT-4o with multi-pass section-finding heuristics
3. **Computes** credit metrics deterministically in Python: EBITDA, leverage (Debt/EBITDA), FCC, free cash flow, Altman Z-Score, and PD score (1-12 scale)
4. **Analyses** the MD&A section using Claude Sonnet to extract segment-level revenue drivers, volume/price splits, and offsetting factors
5. **Generates** a full underwriting memo in markdown
6. **Exports** on demand:
   - Excel workbook (3 sheets: Financial Summary, YoY Bridge, Extraction Notes)
   - Word document (narrative-first analyst working doc with YoY tables, driver bullets, segment analysis, revolver and liquidity, and covenant compliance)
7. **Learns** from analyst corrections via a company-specific profile system with auto-generated extraction rules

---

## Screenshot

![Credit Agent Analysis Results](screenshot.png)

## Architecture

```
Frontend (React + Vite)
│
└── POST /v1/analyze ──────────────────────────────────── FastAPI (main.py)
                                                               │
                          ┌────────────────────────────────────┤
                          │                                    │
                   agents.py                          profiles.py
                   ├── HTML/PDF → text                ├── Company profiles
                   ├── Section finding                ├── Correction store
                   ├── GPT-4o extraction              ├── Rule injection
                   ├── Claude MD&A analysis           └── auto_rule.py
                   ├── Revolver/lease parsing                  │
                   └── validate_and_compute()         pattern_miner.py
                          │                           (cross-company learning)
                   metrics.py
                   ├── compute_ebitda()
                   ├── compute_fcc()
                   ├── compute_leverage()
                   └── compute_altman_z()
                          │
              ┌───────────┴───────────┐
              │                       │
       doc_builder.py         segment_parser.py
       (Excel via openpyxl)   (Claude → segment JSON)
                                       │
                              build_narrative_doc_v2.js
                              (Word via Node.js/docx)
                                       │
                              run_store.py (SQLite)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+, FastAPI, Uvicorn |
| LLM — Extraction | OpenAI GPT-4o |
| LLM — MD&A / Segments | Anthropic Claude Sonnet |
| Document generation | openpyxl (Excel), Node.js + docx (Word) |
| Persistence | SQLite (WAL mode) |
| Frontend | React 18, Vite, ReactMarkdown |
| Testing | pytest (67 tests) |

---

## Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key
- Anthropic API key

### Backend

```bash
cd credit-ai-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm install          # installs docx for Word export
cp .env.example .env # add your API keys
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd credit-ai-frontend
npm install
npm run dev          # runs on localhost:3000
```

---

## Key files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — all endpoints |
| `agents.py` | Extraction pipeline (1,800+ lines) |
| `metrics.py` | Deterministic credit metric functions |
| `credit_scoring.py` | Altman Z-Score, PD score (1–12) |
| `profiles.py` | Company-specific learning / corrections |
| `auto_rule.py` | Auto-generates extraction rules from analyst corrections |
| `pattern_miner.py` | Cross-company pattern learning |
| `doc_builder.py` | Excel workbook builder |
| `segment_parser.py` | MD&A → structured segment JSON |
| `build_narrative_doc_v2.js` | Word document builder |
| `run_store.py` | SQLite persistence with schema migration |
| `tests/test_metrics.py` | 67-test pytest suite |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/analyze` | Upload and analyse a 10-K filing |
| POST | `/v1/export/docs/excel` | Generate Excel from completed run |
| POST | `/v1/export/docs/word` | Generate Word doc from completed run |
| GET | `/v1/profiles/` | List all company profiles |
| POST | `/v1/profiles/{company}/corrections` | Save a field correction |
| POST | `/v1/profiles/{company}/rules` | Save an extraction rule |
| POST | `/v1/profiles/{company}/suggest_rule` | Auto-generate rule from correction |
| GET | `/health` | Health check |

---

## Learning system

The extractor improves over time through three layers:

1. **Company profiles** — analyst corrections saved per company, applied on every future run
2. **Auto-rule generation** — when an analyst corrects a value, Claude analyses the filing context and suggests a precise extraction rule (where to look) for approval
3. **Cross-company pattern mining** — `pattern_miner.py` analyses corrections across all companies to find systematic mistakes and generates general guidance injected into the base extraction prompt

---

## Testing

```bash
cd credit-ai-backend
pytest tests/test_metrics.py -v
```

67 tests covering all credit metric functions, sign convention handling, None propagation, covenant breach detection, and a full realistic CHD scenario.

---

    """## Environment variables

Set these in Railway (production) or a local `.env` file (development). Never commit actual values to git.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key — used for financial extraction and memo generation |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key — used for MD&A analysis and segment parsing |
| `ACCESS_CODE` | Recommended | Password gate for the frontend demo |"""
)

# reconnected

