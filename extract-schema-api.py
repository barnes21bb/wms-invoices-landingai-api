import json
import requests

VA_API_KEY = "YOUR_VA_API_KEY"  # Replace with your API key
headers = {"Authorization": f"Basic {VA_API_KEY}"}
url = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"

base_pdf_path = "your_pdf_path"  # Replace with the path to the file
pdf_name = "filename.pdf"  # Replace the file
pdf_path = f"{base_pdf_path}/{pdf_name}"

# Define your schema
schema = {
    "type": "object",
    "title": "Waste Management Invoice Field Extraction Schema",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "required": [
        "customer_id",
        "customer_name",
        "service_period",
        "invoice_date",
        "invoice_number",
        "current_invoice_charges",
        "vendor_name",
        "vendor_address",
        "vendor_phone",
        "remit_to_address",
        "service_location_address",
        "line_items",
        "from_email_address",
        "to_email_address",
        "email_date",
        "email_subject",
        "email_body_content",
        "gl_account_code",
        "tax_code",
    ],
    "properties": {
        "customer_id": {
            "type": "string",
            "title": "Customer ID",
            "description": "The unique identifier assigned to the customer by the vendor.",
        },
        "customer_name": {
            "type": "string",
            "title": "Customer Name",
            "description": "The full name of the customer or customer organization.",
        },
        "service_period": {
            "type": "string",
            "title": "Service Period",
            "description": "The date range during which the services were provided.",
        },
        "invoice_date": {
            "type": "string",
            "title": "Invoice Date",
            "format": "YYYY-MM-DD",
            "description": "The date the invoice was issued.",
        },
        "invoice_number": {
            "type": "string",
            "title": "Invoice Number",
            "description": "The unique number assigned to the invoice.",
        },
        "current_invoice_charges": {
            "type": "number",
            "title": "Current Invoice Charges",
            "description": "The total amount charged for the current invoice period.",
        },
        "vendor_name": {
            "type": "string",
            "title": "Vendor Name",
            "description": "The name of the vendor or service provider issuing the invoice.",
        },
        "vendor_address": {
            "type": "string",
            "title": "Vendor Address",
            "description": "The mailing address of the vendor or service provider.",
        },
        "vendor_phone": {
            "type": "string",
            "title": "Vendor Phone",
            "description": "The primary phone number(s) for the vendor or service provider.",
        },
        "remit_to_address": {
            "type": "string",
            "title": "Remit To Address",
            "description": "The address to which payments should be sent.",
        },
        "service_location_address": {
            "type": "string",
            "title": "Service Location Address",
            "description": "The address where the service was provided.",
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "title": "Line Item",
                "required": [
                    "description",
                    "date",
                    "ticket_number",
                    "quantity",
                    "amount",
                ],
                "properties": {
                    "description": {
                        "type": "string",
                        "title": "Line Item Description",
                        "description": "A description of the service or product billed.",
                    },
                    "date": {
                        "type": "string",
                        "title": "Line Item Date",
                        "format": "YYYY-MM-DD",
                        "description": "The date associated with the line item (e.g., service date).",
                    },
                    "ticket_number": {
                        "type": "string",
                        "title": "Line Item Ticket Number",
                        "description": "The ticket or reference number for the line item, if available.",
                    },
                    "quantity": {
                        "type": "number",
                        "title": "Line Item Quantity",
                        "description": "The quantity of the service or product billed.",
                    },
                    "amount": {
                        "type": "number",
                        "title": "Line Item Amount",
                        "description": "The monetary amount for the line item.",
                    },
                },
                "description": "Details of a single line item on the invoice.",
            },
            "title": "Line Items",
            "description": "A list of line items detailing services or products billed on the invoice.",
        },
        "from_email_address": {
            "type": "string",
            "title": "From Email Address",
            "description": "The email address of the sender in related correspondence.",
        },
        "to_email_address": {
            "type": "string",
            "title": "To Email Address",
            "description": "The email address of the recipient in related correspondence.",
        },
        "email_date": {
            "type": "string",
            "title": "Email Date",
            "format": "YYYY-MM-DD",
            "description": "The date and time the email was sent.",
        },
        "email_subject": {
            "type": "string",
            "title": "Email Subject",
            "description": "The subject line of the email.",
        },
        "email_body_content": {
            "type": "string",
            "title": "Email Body Content",
            "description": "The main content or body of the email message.",
        },
        "gl_account_code": {
            "type": "string",
            "description": 'General ledger account code annotated on invoice in red text. This string will be similar in format to: "01.294.935.6428.0000".',
        },
        "tax_code": {
            "type": "string",
            "description": "Three or four digit tax code annotated on the invoice in red text. Example tax code are: ABO, ABOF, ACV, EEE, VVV.",
        },
    },
    "description": "A schema for extracting key invoice, vendor, service, line item, and email fields from a markdown document representing a Waste Management invoice and related correspondence.",
}

files = [
    ("pdf", (pdf_name, open(pdf_path, "rb"), "application/pdf")),
]

payload = {"fields_schema": json.dumps(schema)}

response = requests.request("POST", url, headers=headers, files=files, data=payload)

output_data = response.json()["data"]
extracted_info = output_data["extracted_schema"]
print(extracted_info)
