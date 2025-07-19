# WMS Invoices LandingAI API - Project Context

## Project Overview
This is an automated invoice processing system that uses LandingAI's Agentic Document Extraction API to extract structured data from Waste Management invoices and related email correspondence.

## Key Components

### Core Files
- `batch_processor.py` - Main batch processing engine with Pydantic schemas
- `app.py` - Streamlit web interface for file upload and processing
- `main.py` - Basic setup validation script
- `extract-schema-library.py` - Library approach example (recommended)
- `extract-schema-api.py` - Direct API approach example

### Configuration
- `.env` - Contains `VISION_AGENT_API_KEY` for LandingAI API access
- `requirements.txt` - Python dependencies
- `dev-requirements.txt` - Development and testing dependencies

### Data Structure
The system extracts these fields from Waste Management invoices:
- Customer information (ID, name)
- Invoice details (number, date, charges)
- Vendor information (name, address, phone)
- Line items (description, date, ticket number, quantity, amount)
- Email correspondence details
- Accounting codes (GL account code, tax code)

## Architecture Decisions

### Why Library Approach Over Direct API
We chose the `agentic-doc` library approach because:
- Type safety with Pydantic models
- Better error handling
- Built-in batch processing
- Easier testing and maintenance
- IDE support and auto-completion

### Schema Design
- `WasteManagementInvoiceSchema` - Main invoice schema
- `LineItem` - Nested schema for individual invoice line items
- All fields use descriptive titles and detailed descriptions for better AI extraction

## Development Workflow

### Local Development
```bash
# Activate environment
source venv/bin/activate

# Run web app
streamlit run app.py

# Run batch processing
python batch_processor.py

# Run tests
pytest

# Format code
black .
```

### Testing
- Unit tests in `tests/` directory
- Mock LandingAI API calls for reliable testing
- Test both successful and failure scenarios
- Schema validation testing

## Common Tasks

### Processing New Invoices
1. Place PDF files in a directory
2. Use `BatchProcessor.process_multiple_files()`
3. Review results and summaries
4. Export data as needed

### Updating Schemas
1. Modify Pydantic models in `batch_processor.py`
2. Update tests in `tests/test_batch_processor.py`
3. Run validation tests
4. Update documentation

### Troubleshooting
- Check API key is set correctly
- Verify file paths are accessible
- Review LandingAI API documentation for schema requirements
- Check logs for processing errors

## API Limitations & Considerations
- LandingAI API has rate limits
- Large files may take several minutes to process
- API key must be kept secure
- Processing costs apply per document

## Deployment Notes
- GitHub Actions handle CI/CD automatically
- Secrets are configured in GitHub repository
- Docker support available for containerized deployment
- Streamlit app can be deployed to various platforms

## File Organization
- `wms-invoice-pdfs/` - Sample invoice files for testing
- `results/` - Output directory for batch processing results
- `.github/workflows/` - CI/CD automation
- `tests/` - Test suite

## Future Enhancements
- Database integration for storing results
- Excel/CSV export functionality
- Email notification system
- Web dashboard for analytics
- Automated file monitoring and processing