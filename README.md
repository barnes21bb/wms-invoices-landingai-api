# Waste Management Invoice Processing with LandingAI

Automated invoice processing system using LandingAI's Agentic Document Extraction API to extract structured data from Waste Management invoices and related correspondence.

## Features

- 🔍 **Intelligent Document Parsing**: Extract structured data from PDF invoices and emails
- 🌐 **Web Interface**: Streamlit app for easy file upload and result review
- ⚡ **Batch Processing**: Process multiple documents efficiently
- 🛡️ **Type Safety**: Pydantic models for data validation
- 📊 **Financial Analytics**: Automatic calculation of totals and summaries
- 💾 **Export Options**: Save results as JSON or Markdown

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Get your API key from [LandingAI Settings](https://va.landing.ai/settings/api-key) and add to `.env`:

```bash
VISION_AGENT_API_KEY=your-api-key-here
```

### 3. Run Applications

**Streamlit Web App:**
```bash
streamlit run app.py
```

**Batch Processing:**
```bash
python batch_processor.py
```

**Test Setup:**
```bash
python main.py
```

## Project Structure

```
├── app.py                    # Streamlit web application
├── batch_processor.py        # Batch processing with Pydantic schemas
├── main.py                   # Basic setup and API test
├── extract-schema-library.py # Library approach example
├── extract-schema-api.py     # Direct API approach example
├── extract-results.json      # Sample extraction output
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Extracted Data Fields

The system extracts the following fields from Waste Management invoices:

### Invoice Information
- Customer ID and Name
- Invoice Number and Date
- Service Period
- Current Invoice Charges

### Vendor Details
- Vendor Name, Address, Phone
- Remit To Address
- Service Location Address

### Line Items
- Description, Date, Ticket Number
- Quantity and Amount

### Email Correspondence
- From/To Email Addresses
- Email Date and Subject
- Email Body Content

### Accounting Codes
- GL Account Code
- Tax Code

## Usage Examples

### Single File Processing
```python
from batch_processor import BatchProcessor

processor = BatchProcessor()
result = processor.process_single_file("invoice.pdf")
print(result["extraction"])
```

### Batch Processing
```python
processor = BatchProcessor()
file_paths = ["invoice1.pdf", "invoice2.pdf"]
results = processor.process_multiple_files(file_paths)
processor.save_results(results, "batch_results.json")
```

### Streamlit Web Interface
```bash
# Launch web app
streamlit run app.py

# Navigate to http://localhost:8501
# Upload files and view extracted results
```

## Configuration

The system uses `ParseConfig` for centralized settings:

```python
config = ParseConfig(
    api_key="your-api-key",
    extraction_model=WasteManagementInvoiceSchema,
    include_marginalia=False,
    split_size=10,
    extraction_split_size=50
)
```

## Output Format

Results include both clean extracted data and metadata with chunk references:

```json
{
  "extraction": {
    "customer_name": "SIMMONS PREPARED FOODS",
    "invoice_number": "5260154-0592-5",
    "current_invoice_charges": 5346.96,
    "line_items": [...]
  },
  "metadata": {
    "customer_name": {
      "value": "SIMMONS PREPARED FOODS",
      "chunk_references": ["uuid1", "uuid2"]
    }
  }
}
```

## API Documentation

- [LandingAI Quickstart](https://docs.landing.ai/ade/ade-quickstart)
- [Parse Configuration](https://docs.landing.ai/ade/ade-parseconfig)
- [Batch Processing](https://docs.landing.ai/ade/ade-parse-docs)
- [Troubleshooting](https://docs.landing.ai/ade/ade-extract-troubleshoot)

## Requirements

- Python 3.9+
- LandingAI API Key
- Virtual environment recommended