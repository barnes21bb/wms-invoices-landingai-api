# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Automated invoice processing system using LandingAI's Agentic Document Extraction API to extract structured data from Waste Management invoices and email correspondence. Includes comprehensive OCR evaluation tools for measuring accuracy and building ground truth datasets.

**Repository**: https://github.com/simmonsfoods/wm-invoices-landingai-api

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

### Git Commands
```bash
# Clone repository
git clone https://github.com/simmonsfoods/wm-invoices-landingai-api.git

# Push changes (requires authentication setup)
git push simfoods main

# Pull latest changes
git pull simfoods main
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
- `app.py` - Streamlit web interface with 4 main pages:
  - Process New Invoice: Upload and process documents
  - Invoice Summary: Browse processed invoices with filtering
  - Ground Truth Annotation: Create evaluation datasets with PDF viewer
  - Evaluation Dashboard: Compare OCR tool accuracy
- `evaluation_models.py` - Pydantic models for OCR evaluation and ground truth datasets

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

## File Organization

### Results Directory Structure
```
results/
├── processed_invoices/
│   ├── landingai-parse/     # Parse function results ($0.03/page)
│   │   ├── *_result.json    # Raw document analysis (50-90KB)
│   │   └── README.md        # Documentation and cost info
│   └── landingai-extract/   # Extract function results ($0.03/page)
│       ├── *_extraction.json # Structured data (1-4KB)
│       └── README.md        # Documentation and cost info
├── uploaded_files/          # Original PDF files
├── evaluation_datasets/     # OCR evaluation datasets
│   ├── main.json           # Primary evaluation dataset
│   ├── test.json           # Testing dataset
│   └── validation.json     # Validation dataset
```

### Cost Structure
- **LandingAI Parse**: $0.03 per page (document analysis and markdown)
- **LandingAI Extract**: $0.03 per page (structured data extraction)
- **Total per invoice**: $0.06 per page for complete processing pipeline

## Deployment and Collaboration

### GitHub Authentication Setup
For team members working with this repository:

1. **Personal Access Token** (Recommended):
   - Create token at https://github.com/settings/tokens
   - Required scopes: `repo`, `workflow`
   - Authorize for SSO with simmonsfoods organization
   - Configure git credential storage: `git config --global credential.helper store`

2. **SSH Key Alternative**:
   - Add SSH key to GitHub account (not repository-specific)
   - Authorize for SSO with simmonsfoods organization

### Team Workflow
- **Main Branch**: Production-ready code
- **Feature Branches**: Use for development
- **Pull Requests**: Required for code review
- **CI/CD**: GitHub Actions workflows for testing and quality checks

## Important Implementation Notes
- Always use the library approach (`agentic-doc`) over direct API calls for better error handling
- Results include both extracted data and metadata with chunk references for verification
- The system handles rate limiting and large file processing automatically
- PDF files are excluded from git repository for security (see .gitignore)