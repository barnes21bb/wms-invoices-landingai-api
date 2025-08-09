# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Automated invoice processing system using LandingAI's Agentic Document Extraction API to extract structured data from Waste Management invoices and email correspondence.

## Essential Commands

### Setup and Environment
```bash
# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

### Development Commands
```bash
# Run web application
streamlit run app.py

# Run batch processing
python batch_processor.py

# Validate setup
python main.py
```

### Testing and Quality
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run single test file
pytest tests/test_batch_processor.py

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy . --ignore-missing-imports

# Security scanning
bandit -r .
safety check
```

## Architecture Overview

### Core Processing Pipeline
The system follows a three-layer architecture:
1. **Input Layer**: PDF/email document ingestion via Streamlit UI or batch processing
2. **Processing Layer**: LandingAI `agentic-doc` library with Pydantic schema validation
3. **Output Layer**: Structured JSON/Markdown export with metadata and chunk references

### Key Components
- `batch_processor.py:BatchProcessor` - Main processing engine with type-safe Pydantic schemas
- `batch_processor.py:WasteManagementInvoiceSchema` - Primary data model for invoice extraction
- `batch_processor.py:LineItem` - Nested model for invoice line items
- `app.py` - Streamlit web interface for interactive processing

### Schema Design Philosophy
All Pydantic models use descriptive Field titles and descriptions to guide LandingAI's extraction. The `WasteManagementInvoiceSchema` captures:
- Customer/vendor information
- Invoice metadata (number, date, charges)
- Line items with quantities and amounts
- Email correspondence details
- Accounting codes (GL account, tax codes)

### Processing Patterns
- **Library Approach**: Uses `agentic-doc.parse()` with `ParseConfig` for type safety and error handling
- **Batch Processing**: `BatchProcessor.process_multiple_files()` handles multiple documents with progress tracking
- **Result Structure**: Returns both clean extracted data and metadata with chunk references for verification

## Configuration Requirements
- `.env` file with `VISION_AGENT_API_KEY` from LandingAI Settings
- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- Virtual environment recommended

## Testing Strategy
- Unit tests mock LandingAI API calls for reliability
- Schema validation tests ensure Pydantic models work correctly
- CI pipeline includes security scanning with bandit and safety
- Coverage reporting with pytest-cov

## Important Implementation Notes
- Always use the library approach (`agentic-doc`) over direct API calls for better error handling
- Results include both extracted data and metadata with chunk references for verification
- The system handles rate limiting and large file processing automatically
- Sample invoices in `wms-invoice-pdfs/` directory for testing