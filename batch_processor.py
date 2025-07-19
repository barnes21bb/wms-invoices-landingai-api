#!/usr/bin/env python3
"""
Batch processor for Waste Management invoices using LandingAI
Improved implementation combining the best of both approaches
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agentic_doc.parse import parse, ParseConfig

# Load environment variables
load_dotenv()

class LineItem(BaseModel):
    description: str = Field(
        ...,
        description='A description of the service or product billed.',
        title='Line Item Description',
    )
    date: str = Field(
        ...,
        description='The date associated with the line item (e.g., service date).',
        title='Line Item Date',
    )
    ticket_number: str = Field(
        ...,
        description='The ticket or reference number for the line item, if available.',
        title='Line Item Ticket Number',
    )
    quantity: float = Field(
        ...,
        description='The quantity of the service or product billed.',
        title='Line Item Quantity',
    )
    amount: float = Field(
        ...,
        description='The monetary amount for the line item.',
        title='Line Item Amount',
    )

class WasteManagementInvoiceSchema(BaseModel):
    customer_id: str = Field(
        ...,
        description='The unique identifier assigned to the customer by the vendor.',
        title='Customer ID',
    )
    customer_name: str = Field(
        ...,
        description='The full name of the customer or customer organization.',
        title='Customer Name',
    )
    service_period: str = Field(
        ...,
        description='The date range during which the services were provided.',
        title='Service Period',
    )
    invoice_date: str = Field(
        ..., description='The date the invoice was issued.', title='Invoice Date'
    )
    invoice_number: str = Field(
        ...,
        description='The unique number assigned to the invoice.',
        title='Invoice Number',
    )
    current_invoice_charges: float = Field(
        ...,
        description='The total amount charged for the current invoice period.',
        title='Current Invoice Charges',
    )
    vendor_name: str = Field(
        ...,
        description='The name of the vendor or service provider issuing the invoice.',
        title='Vendor Name',
    )
    vendor_address: str = Field(
        ...,
        description='The mailing address of the vendor or service provider.',
        title='Vendor Address',
    )
    vendor_phone: str = Field(
        ...,
        description='The primary phone number(s) for the vendor or service provider.',
        title='Vendor Phone',
    )
    remit_to_address: str = Field(
        ...,
        description='The address to which payments should be sent.',
        title='Remit To Address',
    )
    service_location_address: str = Field(
        ...,
        description='The address where the service was provided.',
        title='Service Location Address',
    )
    line_items: List[LineItem] = Field(
        ...,
        description='A list of line items detailing services or products billed on the invoice.',
        title='Line Items',
    )
    from_email_address: str = Field(
        ...,
        description='The email address of the sender in related correspondence.',
        title='From Email Address',
    )
    to_email_address: str = Field(
        ...,
        description='The email address of the recipient in related correspondence.',
        title='To Email Address',
    )
    email_date: str = Field(
        ..., description='The date and time the email was sent.', title='Email Date'
    )
    email_subject: str = Field(
        ..., description='The subject line of the email.', title='Email Subject'
    )
    email_body_content: str = Field(
        ...,
        description='The main content or body of the email message.',
        title='Email Body Content',
    )
    gl_account_code: str = Field(
        ...,
        description='General ledger account code annotated on invoice in red text. This string will be similar in format to: "01.294.935.6428.0000".',
    )
    tax_code: str = Field(
        ...,
        description='Three or four digit tax code annotated on the invoice in red text. Example tax code are: ABO, ABOF, ACV, EEE, VVV.',
    )

class BatchProcessor:
    """Batch processor for waste management invoices"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check API key
        self.api_key = os.getenv('VISION_AGENT_API_KEY')
        if not self.api_key:
            raise ValueError("VISION_AGENT_API_KEY not found in environment variables")
        
        # Configure parser
        self.config = ParseConfig(
            api_key=self.api_key,
            extraction_model=WasteManagementInvoiceSchema,
            include_marginalia=False,
            include_metadata_in_markdown=True,
            split_size=10,
            extraction_split_size=50
        )
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and return extracted data"""
        try:
            results = parse(file_path, config=self.config)
            
            if not results:
                raise ValueError(f"No results returned for {file_path}")
            
            # Get the first result (should be only one for single file)
            result = results[0]
            
            # Extract structured data
            extracted_data = {
                "file_path": file_path,
                "extraction": result.extraction.model_dump() if result.extraction else None,
                "markdown": result.markdown,
                "chunks": result.chunks,
                "success": True,
                "error": None
            }
            
            return extracted_data
            
        except Exception as e:
            return {
                "file_path": file_path,
                "extraction": None,
                "markdown": None,
                "chunks": None,
                "success": False,
                "error": str(e)
            }
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files and return all extracted data"""
        results = []
        
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            result = self.process_single_file(file_path)
            results.append(result)
            
            if result["success"]:
                print(f"✅ Success: {file_path}")
            else:
                print(f"❌ Failed: {file_path} - {result['error']}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_filename: str = "batch_results.json"):
        """Save results to JSON file"""
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate processing summary"""
        total_files = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total_files - successful
        
        total_charges = 0
        processed_invoices = []
        
        for result in results:
            if result["success"] and result["extraction"]:
                extraction = result["extraction"]
                if extraction.get("current_invoice_charges"):
                    total_charges += extraction["current_invoice_charges"]
                processed_invoices.append({
                    "file": Path(result["file_path"]).name,
                    "invoice_number": extraction.get("invoice_number"),
                    "customer_name": extraction.get("customer_name"),
                    "charges": extraction.get("current_invoice_charges"),
                    "line_items_count": len(extraction.get("line_items", []))
                })
        
        return {
            "processing_summary": {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "success_rate": f"{(successful/total_files)*100:.1f}%" if total_files > 0 else "0%"
            },
            "financial_summary": {
                "total_charges": total_charges,
                "processed_invoices": processed_invoices
            },
            "failed_files": [r["file_path"] for r in results if not r["success"]]
        }

def main():
    """Example usage"""
    processor = BatchProcessor()
    
    # Example: Process files from a directory
    # file_paths = list(Path("documents").glob("*.pdf"))
    
    # Example: Process specific files
    file_paths = [
        "path/to/invoice1.pdf",
        "path/to/invoice2.pdf"
    ]
    
    print("Starting batch processing...")
    results = processor.process_multiple_files([str(p) for p in file_paths])
    
    # Save results
    output_path = processor.save_results(results)
    
    # Generate and save summary
    summary = processor.generate_summary(results)
    summary_path = processor.save_results([summary], "processing_summary.json")
    
    print(f"\nProcessing complete!")
    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Total files: {summary['processing_summary']['total_files']}")
    print(f"- Successful: {summary['processing_summary']['successful']}")
    print(f"- Failed: {summary['processing_summary']['failed']}")
    print(f"- Success rate: {summary['processing_summary']['success_rate']}")
    print(f"- Total charges: ${summary['financial_summary']['total_charges']:,.2f}")

if __name__ == "__main__":
    main()