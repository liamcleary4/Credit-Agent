
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
Analyst Input (3 modes)
│
├── Company name search ────────────────────────────── edgar_client.py
│   GET /v1/edgar/search                               ├── SEC EDGAR company registry
│   GET /v1/edgar/{cik}/filings                        ├── Submissions API
│   POST /v1/analyze/edgar                             └── Filing document fetch
│
├── File upload ─────────────────────────────────────── main.py (file_to_text)
│   POST /v1/analyze                                    ├── PDF parser (pypdf)
│                                                       └── HTML parser (BeautifulSoup)
│
└── URL paste
    POST /v1/analyze/url ──────────────────────────── filing_fetcher.py
                                                        └── HTTP fetch + same pipeline
                          │
                          ▼
                   FastAPI (main.py)
                          │
            ┌─────────────┼─────────────┐
            │             │             │
       agents.py    profiles.py    run_store.py
       ├── Section    ├── Corrections  ├── SQLite cache
       │   finding    ├── Rule inject  └── Export store
       ├── GPT-4o     └── auto_rule.py
       │   extraction
       ├── Claude
       │   MD&A
       └── validate_
           and_compute()
                │
           metrics.py
           ├── compute_ebitda()
           ├── compute_fcc()
           ├── compute_leverage()
           └── compute_altman_z()
                │
     ┌──────────┴──────────┐
     │                     │
doc_builder.py      segment_parser.py
(Excel/openpyxl)    (Claude → JSON)
                           │
                  build_narrative_doc_v2.js
                  (Word via Node.js/docx)
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
| Testing | pytest (68 tests) |

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
| `tests/test_metrics.py` | 68-test pytest suite |

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

The extractor improves over time through two layers:

1. **Company profiles** — analyst corrections saved per company, applied automatically on every future run for that company
2. **Auto-rule generation** — when an analyst corrects a value, Claude analyses where the extractor originally found the wrong number and suggests a precise extraction rule for approval. Approved rules are injected into the GPT-4o prompt on the next run so the extractor finds the right number dynamically rather than relying on a hardcoded value

---

---

## Testing

```bash
cd credit-ai-backend
pytest tests/test_metrics.py -v
```

68 tests covering all credit metric functions, sign convention handling, None propagation, covenant breach detection, and a full realistic CHD scenario.

---

## Environment variables

Set these in Railway (production) or a local `.env` file (development). Never commit actual values to git.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key used for financial extraction and memo generation |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key used for MD&A analysis and segment parsing |
| `ACCESS_CODE` | Recommended | Password gate for the frontend demo |

---

## Live demo

[credit-ai-frontend.vercel.app](https://credit-ai-frontend.vercel.app) — request access code from [Liam Cleary](mailto:ljcleary4@gmail.com)
