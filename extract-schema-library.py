from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field
from agentic_doc.parse import parse


class LineItem(BaseModel):
    description: str = Field(
        ...,
        description="A description of the service or product billed.",
        title="Line Item Description",
    )
    date: str = Field(
        ...,
        description="The date associated with the line item (e.g., service date).",
        title="Line Item Date",
    )
    ticket_number: str = Field(
        ...,
        description="The ticket or reference number for the line item, if available.",
        title="Line Item Ticket Number",
    )
    quantity: float = Field(
        ...,
        description="The quantity of the service or product billed.",
        title="Line Item Quantity",
    )
    amount: float = Field(
        ...,
        description="The monetary amount for the line item.",
        title="Line Item Amount",
    )


class WasteManagementInvoiceFieldExtractionSchema(BaseModel):
    customer_id: str = Field(
        ...,
        description="The unique identifier assigned to the customer by the vendor.",
        title="Customer ID",
    )
    customer_name: str = Field(
        ...,
        description="The full name of the customer or customer organization.",
        title="Customer Name",
    )
    service_period: str = Field(
        ...,
        description="The date range during which the services were provided.",
        title="Service Period",
    )
    invoice_date: str = Field(
        ..., description="The date the invoice was issued.", title="Invoice Date"
    )
    invoice_number: str = Field(
        ...,
        description="The unique number assigned to the invoice.",
        title="Invoice Number",
    )
    current_invoice_charges: float = Field(
        ...,
        description="The total amount charged for the current invoice period.",
        title="Current Invoice Charges",
    )
    vendor_name: str = Field(
        ...,
        description="The name of the vendor or service provider issuing the invoice.",
        title="Vendor Name",
    )
    vendor_address: str = Field(
        ...,
        description="The mailing address of the vendor or service provider.",
        title="Vendor Address",
    )
    vendor_phone: str = Field(
        ...,
        description="The primary phone number(s) for the vendor or service provider.",
        title="Vendor Phone",
    )
    remit_to_address: str = Field(
        ...,
        description="The address to which payments should be sent.",
        title="Remit To Address",
    )
    service_location_address: str = Field(
        ...,
        description="The address where the service was provided.",
        title="Service Location Address",
    )
    line_items: List[LineItem] = Field(
        ...,
        description="A list of line items detailing services or products billed on the invoice.",
        title="Line Items",
    )
    from_email_address: str = Field(
        ...,
        description="The email address of the sender in related correspondence.",
        title="From Email Address",
    )
    to_email_address: str = Field(
        ...,
        description="The email address of the recipient in related correspondence.",
        title="To Email Address",
    )
    email_date: str = Field(
        ..., description="The date and time the email was sent.", title="Email Date"
    )
    email_subject: str = Field(
        ..., description="The subject line of the email.", title="Email Subject"
    )
    email_body_content: str = Field(
        ...,
        description="The main content or body of the email message.",
        title="Email Body Content",
    )
    gl_account_code: str = Field(
        ...,
        description='General ledger account code annotated on invoice in red text. This string will be similar in format to: "01.294.935.6428.0000".',
    )
    tax_code: str = Field(
        ...,
        description="Three or four digit tax code annotated on the invoice in red text. Example tax code are: ABO, ABOF, ACV, EEE, VVV.",
    )


# Parse a file and extract the fields
results = parse(
    "mydoc.pdf", extraction_model=WasteManagementInvoiceFieldExtractionSchema
)
fields = results[0].extraction

# Return the value of the extracted fields
print(fields)
