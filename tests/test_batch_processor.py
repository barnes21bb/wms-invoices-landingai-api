import pytest
from unittest.mock import Mock, patch
from batch_processor import BatchProcessor, WasteManagementInvoiceSchema, LineItem


def test_line_item_validation():
    """Test LineItem schema validation"""
    line_item_data = {
        "description": "40 YD COMPACTOR",
        "date": "2024-12-16",
        "ticket_number": "664193",
        "quantity": 1.0,
        "amount": 480.26,
    }

    line_item = LineItem(**line_item_data)
    assert line_item.description == "40 YD COMPACTOR"
    assert line_item.amount == 480.26


def test_invoice_schema_validation():
    """Test WasteManagementInvoiceSchema validation"""
    invoice_data = {
        "customer_id": "8-08612-13004",
        "customer_name": "SIMMONS PREPARED FOODS",
        "service_period": "2024-12-16 to 2024-12-31",
        "invoice_date": "2025-01-02",
        "invoice_number": "5260154-0592-5",
        "current_invoice_charges": 5346.96,
        "vendor_name": "WASTE MANAGEMENT OF ARKANSAS, INC.",
        "vendor_address": "PO BOX 3020, MONROE, WI 53566-8320",
        "vendor_phone": "(800) 607-9509",
        "remit_to_address": "WM CORPORATE SERVICES, INC.",
        "service_location_address": "2101 Twin Circle Dr, Van Buren AR 72956",
        "line_items": [
            {
                "description": "40 YD COMPACTOR",
                "date": "2024-12-16",
                "ticket_number": "664193",
                "quantity": 1.0,
                "amount": 480.26,
            }
        ],
        "from_email_address": "test@example.com",
        "to_email_address": "recipient@example.com",
        "email_date": "2025-01-08T10:29:00",
        "email_subject": "Test Invoice",
        "email_body_content": "Test content",
        "gl_account_code": "02.385.309.6419.0000",
        "tax_code": "ACV",
    }

    invoice = WasteManagementInvoiceSchema(**invoice_data)
    assert invoice.customer_name == "SIMMONS PREPARED FOODS"
    assert invoice.current_invoice_charges == 5346.96
    assert len(invoice.line_items) == 1


@patch.dict("os.environ", {"VISION_AGENT_API_KEY": "test-key"})
def test_batch_processor_init():
    """Test BatchProcessor initialization"""
    processor = BatchProcessor(output_dir="test_results")
    assert processor.api_key == "test-key"
    assert processor.output_dir.name == "test_results"


def test_batch_processor_no_api_key():
    """Test BatchProcessor fails without API key"""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="VISION_AGENT_API_KEY not found"):
            BatchProcessor()


@patch("batch_processor.parse")
@patch.dict("os.environ", {"VISION_AGENT_API_KEY": "test-key"})
def test_process_single_file_success(mock_parse):
    """Test successful single file processing"""
    # Mock the parse result
    mock_result = Mock()
    mock_result.extraction = Mock()
    mock_result.extraction.model_dump.return_value = {"test": "data"}
    mock_result.markdown = "# Test markdown"
    mock_result.chunks = [{"chunk": "data"}]
    mock_parse.return_value = [mock_result]

    processor = BatchProcessor()
    result = processor.process_single_file("test.pdf")

    assert result["success"] is True
    assert result["extraction"] == {"test": "data"}
    assert result["markdown"] == "# Test markdown"
    assert result["error"] is None


@patch("batch_processor.parse")
@patch.dict("os.environ", {"VISION_AGENT_API_KEY": "test-key"})
def test_process_single_file_failure(mock_parse):
    """Test failed single file processing"""
    mock_parse.side_effect = Exception("API Error")

    processor = BatchProcessor()
    result = processor.process_single_file("test.pdf")

    assert result["success"] is False
    assert result["extraction"] is None
    assert "API Error" in result["error"]


def test_generate_summary():
    """Test processing summary generation"""
    results = [
        {
            "success": True,
            "file_path": "invoice1.pdf",
            "extraction": {
                "invoice_number": "INV-001",
                "customer_name": "Customer 1",
                "current_invoice_charges": 1000.0,
                "line_items": [{"item": 1}, {"item": 2}],
            },
        },
        {"success": False, "file_path": "invoice2.pdf", "extraction": None},
        {
            "success": True,
            "file_path": "invoice3.pdf",
            "extraction": {
                "invoice_number": "INV-002",
                "customer_name": "Customer 2",
                "current_invoice_charges": 2000.0,
                "line_items": [{"item": 1}],
            },
        },
    ]

    processor = BatchProcessor.__new__(BatchProcessor)  # Create without __init__
    summary = processor.generate_summary(results)

    assert summary["processing_summary"]["total_files"] == 3
    assert summary["processing_summary"]["successful"] == 2
    assert summary["processing_summary"]["failed"] == 1
    assert summary["processing_summary"]["success_rate"] == "66.7%"
    assert summary["financial_summary"]["total_charges"] == 3000.0
    assert len(summary["financial_summary"]["processed_invoices"]) == 2
    assert len(summary["failed_files"]) == 1
