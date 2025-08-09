import streamlit as st
import os
import tempfile
import json
import hashlib
import shutil
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
from agentic_doc.parse import parse
from batch_processor import BatchProcessor, WasteManagementInvoiceSchema, LineItem
import PyPDF2
from PIL import Image

# Load environment variables
load_dotenv()

# Results directories
RESULTS_DIR = Path("results")
PROCESSED_DIR = RESULTS_DIR / "processed_invoices"
UPLOADED_DIR = RESULTS_DIR / "uploaded_files"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_DIR.mkdir(parents=True, exist_ok=True)

# Cost configuration
COST_PER_PAGE = 0.03  # $0.03 per page


def get_file_hash(file_content):
    """Generate SHA-256 hash of file content for duplicate detection"""
    return hashlib.sha256(file_content).hexdigest()


def get_page_count(file_content, file_type):
    """Get the number of pages in a file for cost estimation"""
    try:
        if file_type == "application/pdf":
            # Handle PDF files
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()

                with open(tmp_file.name, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    return len(pdf_reader.pages)
        else:
            # For image files (PNG, JPG, JPEG), assume 1 page
            return 1
    except Exception as e:
        # If we can't determine page count, assume 1 page
        st.warning(f"Could not determine page count, assuming 1 page: {str(e)}")
        return 1


def save_processing_result(filename, file_hash, result, extraction_data=None):
    """Save processing results to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(filename).stem

    # Create result metadata
    metadata = {
        "filename": filename,
        "file_hash": file_hash,
        "processed_at": timestamp,
        "has_extraction": extraction_data is not None,
    }

    # Extract markdown and chunks from the result properly
    markdown_content = None
    chunks_data = None

    # Handle different types of results
    if hasattr(result, "markdown"):
        markdown_content = result.markdown
    elif hasattr(result, "__iter__") and len(result) > 0:
        # Handle ParsedDocument list
        first_doc = result[0] if isinstance(result, list) else result
        if hasattr(first_doc, "markdown"):
            markdown_content = first_doc.markdown
        if hasattr(first_doc, "chunks"):
            # Convert chunks to serializable format
            chunks_data = []
            for chunk in first_doc.chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "chunk_type": (
                        str(chunk.chunk_type) if hasattr(chunk, "chunk_type") else None
                    ),
                    "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else None,
                }
                chunks_data.append(chunk_dict)

    # Save main result
    result_file = PROCESSED_DIR / f"{base_name}_{timestamp}_result.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "raw_result": str(result),
                "markdown": markdown_content,
                "chunks": chunks_data,
            },
            f,
            indent=2,
        )

    # Save structured extraction if available
    if extraction_data:
        extraction_file = PROCESSED_DIR / f"{base_name}_{timestamp}_extraction.json"
        with open(extraction_file, "w") as f:
            json.dump(extraction_data, f, indent=2)

    return result_file


def check_duplicate_file(file_content):
    """Check if file has been processed before based on hash"""
    file_hash = get_file_hash(file_content)

    # Check existing results for this hash
    for result_file in PROCESSED_DIR.glob("*_result.json"):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                if data.get("metadata", {}).get("file_hash") == file_hash:
                    return True, result_file
        except:
            continue

    return False, None


def load_all_results():
    """Load all processing results for browsing"""
    results = []

    for result_file in sorted(PROCESSED_DIR.glob("*_result.json"), reverse=True):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

                # Try to load corresponding extraction file
                # Convert "file_result.json" to "file_extraction.json"
                extraction_file_name = result_file.name.replace(
                    "_result.json", "_extraction.json"
                )
                extraction_file = PROCESSED_DIR / extraction_file_name
                extraction_data = None

                if extraction_file.exists():
                    try:
                        with open(extraction_file, "r") as ef:
                            extraction_data = json.load(ef)
                    except:
                        pass

                # Add extraction data to the main data
                if extraction_data:
                    data["extraction"] = extraction_data

                results.append(
                    {
                        "file_path": result_file,
                        "metadata": data.get("metadata", {}),
                        "data": data,
                    }
                )
        except Exception as e:
            print(f"Error loading {result_file}: {e}")  # Debug output
            continue

    return results


def main():
    st.set_page_config(
        page_title="WMS Invoice Processor", page_icon="📄", layout="wide"
    )

    st.title("📄 WMS Invoice Processor")
    st.markdown(
        "Upload Waste Management invoices and extract structured data using LandingAI's API"
    )

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:", ["Process New Invoice", "Invoice Summary"]
    )

    if page == "Process New Invoice":
        process_invoice_page()
    else:
        invoice_summary_page()


def process_invoice_page():

    st.header("📤 Upload New Invoice")

    # Check API key
    api_key = os.getenv("VISION_AGENT_API_KEY")
    if not api_key:
        st.error("❌ VISION_AGENT_API_KEY not found in environment variables")
        st.info(
            "Please set your API key in the .env file or as an environment variable"
        )
        return

    st.success("✅ API Key loaded successfully")

    # File upload - Allow multiple files
    uploaded_files = st.file_uploader(
        "Choose Waste Management invoices to process",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload PDF or image files of Waste Management invoices",
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Process files information and check for duplicates
        file_info = []
        new_files = []
        duplicate_files = []

        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer

            is_duplicate, existing_file = check_duplicate_file(file_content)

            # Get page count for cost estimation
            page_count = get_page_count(file_content, uploaded_file.type)

            info = {
                "file": uploaded_file,
                "content": file_content,
                "hash": get_file_hash(file_content),
                "is_duplicate": is_duplicate,
                "existing_file": existing_file,
                "page_count": page_count,
            }
            file_info.append(info)

            if is_duplicate:
                duplicate_files.append(info)
            else:
                new_files.append(info)

        # Calculate cost estimation
        total_pages = sum([info["page_count"] for info in new_files])
        estimated_cost = total_pages * COST_PER_PAGE

        # Display file summary
        st.subheader("📋 Upload Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(uploaded_files))
        with col2:
            st.metric("New Files", len(new_files))
        with col3:
            st.metric("Duplicates", len(duplicate_files))
        with col4:
            if new_files:
                st.metric("Est. Cost", f"${estimated_cost:.2f}")
            else:
                st.metric("Est. Cost", "$0.00")

        # Show duplicate files warning
        if duplicate_files:
            st.warning(f"⚠️ {len(duplicate_files)} file(s) have already been processed:")
            for dup in duplicate_files:
                st.write(f"• {dup['file'].name}")

        # Show new files to be processed
        if new_files:
            st.success(f"✅ {len(new_files)} new file(s) ready for processing:")

            # Display file details in expandable sections
            for i, info in enumerate(new_files, 1):
                file_cost = info["page_count"] * COST_PER_PAGE
                with st.expander(
                    f"📄 {i}. {info['file'].name} ({info['page_count']} pages - ${file_cost:.2f})"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**File size:** {info['file'].size:,} bytes")
                        st.write(f"**File type:** {info['file'].type}")
                        st.write(f"**Pages:** {info['page_count']}")
                    with col2:
                        st.write(f"**File hash:** {info['hash'][:16]}...")
                        st.write(f"**Processing cost:** ${file_cost:.2f}")
                        st.write(f"**Status:** New file")

            # Cost warning and process button
            if estimated_cost > 1.00:  # Show warning for costs over $1
                st.warning(
                    f"⚠️ Processing these {len(new_files)} file(s) with {total_pages} pages will cost approximately **${estimated_cost:.2f}**"
                )
            else:
                st.info(
                    f"💰 Processing cost: **${estimated_cost:.2f}** ({total_pages} pages × ${COST_PER_PAGE:.2f}/page)"
                )

            # Process button
            if st.button(
                f"🚀 Process {len(new_files)} Invoice(s) - ${estimated_cost:.2f}",
                type="primary",
            ):
                process_multiple_files(new_files)
        else:
            st.info(
                "No new files to process. All uploaded files have already been processed."
            )


def process_multiple_files(file_info_list):
    """Process multiple files sequentially with progress tracking"""

    # Create progress tracking
    total_files = len(file_info_list)
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    processed_results = []

    for i, info in enumerate(file_info_list):
        file = info["file"]
        file_content = info["content"]
        file_hash = info["hash"]

        # Update progress
        progress = (i) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{total_files}: {file.name}")

        try:
            # Save uploaded file permanently
            saved_file_path = UPLOADED_DIR / file.name
            with open(saved_file_path, "wb") as f:
                f.write(file_content)

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{file.name}"
            ) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            # Process with BatchProcessor for structured extraction
            processor = BatchProcessor()
            structured_result = processor.process_single_file(tmp_file_path)

            # Also get raw parse result
            raw_result = parse(tmp_file_path)

            # Clean up temp file
            os.unlink(tmp_file_path)

            # Save results to disk
            extraction_data = (
                structured_result.get("extraction") if structured_result else None
            )
            result_file = save_processing_result(
                file.name, file_hash, raw_result, extraction_data
            )

            # Store result for display
            result_data = {
                "filename": file.name,
                "success": True,
                "result_file": result_file.name,
                "extraction_data": extraction_data,
                "metadata": {
                    "filename": file.name,
                    "file_hash": file_hash,
                    "processed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "has_extraction": extraction_data is not None,
                },
                "raw_result": str(raw_result),
                "markdown": getattr(raw_result, "markdown", None),
                "chunks": getattr(raw_result, "chunks", None),
                "extraction": extraction_data,
            }
            processed_results.append(result_data)

        except Exception as e:
            # Store error result
            result_data = {"filename": file.name, "success": False, "error": str(e)}
            processed_results.append(result_data)

    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed processing {total_files} file(s)")

    # Display results summary
    successful = [r for r in processed_results if r["success"]]
    failed = [r for r in processed_results if not r["success"]]

    st.subheader("🎯 Processing Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Successful", len(successful))
    with col2:
        st.metric("❌ Failed", len(failed))

    # Show failed files
    if failed:
        st.error("❌ Failed to process these files:")
        for result in failed:
            st.write(f"• **{result['filename']}**: {result['error']}")

    # Show successful files
    if successful:
        st.success(f"✅ Successfully processed {len(successful)} file(s)")

        # Display each result in tabs or expandable sections
        if len(successful) == 1:
            # Single file - show full results immediately
            result = successful[0]
            display_single_result(result, result["filename"])
        else:
            # Multiple files - show in expandable sections
            st.markdown("### 📄 Processed Invoices")
            for i, result in enumerate(successful, 1):
                with st.expander(
                    f"📋 {i}. {result['filename']} - ${result['extraction']['current_invoice_charges'] if result['extraction'] else 'N/A'}"
                ):
                    display_single_result(result, result["filename"])


def display_single_result(result_data, filename):
    """Display processing results in a structured format"""
    st.subheader(f"📊 {filename}")

    # Show metadata
    metadata = result_data.get("metadata", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processed At", metadata.get("processed_at", "Unknown"))
    with col2:
        st.metric("Has Extraction", "Yes" if metadata.get("has_extraction") else "No")
    with col3:
        file_hash = metadata.get("file_hash", "Unknown")
        st.metric(
            "File Hash", file_hash[:8] + "..." if len(file_hash) > 8 else file_hash
        )

    # Extract markdown from raw_result string if available
    def extract_markdown_from_raw_result(raw_result_str):
        """Extract markdown content from raw result string"""
        if not raw_result_str:
            return None
        try:
            # Look for markdown= pattern in the raw result string
            import re

            # Match markdown='...' with proper handling of escaped quotes
            markdown_pattern = r"markdown='((?:[^'\\]|\\.)*?)'"
            match = re.search(markdown_pattern, raw_result_str, re.DOTALL)
            if match:
                # Unescape the content
                markdown_content = match.group(1)
                markdown_content = markdown_content.replace("\\'", "'")
                markdown_content = markdown_content.replace("\\n", "\n")
                markdown_content = markdown_content.replace("\\t", "\t")
                return markdown_content
        except Exception as e:
            st.error(f"Error extracting markdown: {str(e)}")
        return None

    # Get markdown content
    markdown_content = result_data.get("markdown")  # Direct from result
    if not markdown_content and result_data.get("raw_result"):
        # Try to extract from raw_result string
        markdown_content = extract_markdown_from_raw_result(
            result_data.get("raw_result")
        )

    # Create tabs for different views
    if result_data.get("extraction"):
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📝 Structured Data", "📄 Markdown", "🔧 Raw JSON", "📋 Summary"]
        )

        with tab1:
            st.markdown("### Extracted Invoice Data")
            extraction = result_data["extraction"]

            # Display key invoice information
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Customer Information:**")
                st.write(f"• Customer ID: {extraction.get('customer_id', 'N/A')}")
                st.write(f"• Customer Name: {extraction.get('customer_name', 'N/A')}")
                st.write(f"• Invoice Number: {extraction.get('invoice_number', 'N/A')}")
                st.write(f"• Invoice Date: {extraction.get('invoice_date', 'N/A')}")

            with col2:
                st.write("**Financial Information:**")
                st.write(
                    f"• Current Charges: ${extraction.get('current_invoice_charges', 'N/A')}"
                )
                st.write(f"• Service Period: {extraction.get('service_period', 'N/A')}")

            # Line items in table format
            if extraction.get("line_items"):
                st.write("**Line Items:**")

                # Create DataFrame from line items with categories and tonnage
                line_items_data = []
                category_totals = {
                    "standard": 0,
                    "minimum_tonnage": 0,
                    "overage": 0,
                    "fees": 0,
                }
                total_invoice_tons = 0

                for i, item in enumerate(extraction["line_items"], 1):
                    # Create LineItem instance to get category and tonnage
                    try:
                        line_item = LineItem(**item)
                        category = line_item.category.value
                        category_display = category.replace("_", " ").title()
                        estimated_tons = line_item.estimated_tons
                        category_totals[category] += item.get("amount", 0)
                        total_invoice_tons += estimated_tons
                    except Exception:
                        category_display = "Unknown"
                        estimated_tons = 0.0

                    line_items_data.append(
                        {
                            "#": i,
                            "Description": item.get("description", "N/A"),
                            "Category": category_display,
                            "Date": item.get("date", "N/A"),
                            "Ticket #": item.get("ticket_number", "N/A"),
                            "Quantity": item.get("quantity", "N/A"),
                            "Est. Tons": f"{estimated_tons:,.2f}" if estimated_tons > 0 else "0.00",
                            "Amount": (
                                f"${item.get('amount', 0):,.2f}"
                                if item.get("amount") != "N/A"
                                else "N/A"
                            ),
                        }
                    )

                if line_items_data:
                    line_items_df = pd.DataFrame(line_items_data)
                    st.dataframe(
                        line_items_df, use_container_width=True, hide_index=True
                    )

                    # Show total and category breakdown
                    total_amount = sum(
                        [
                            item.get("amount", 0)
                            for item in extraction["line_items"]
                            if isinstance(item.get("amount"), (int, float))
                        ]
                    )
                    st.write(f"**Line Items Total: ${total_amount:,.2f} | Total Estimated Tons: {total_invoice_tons:,.2f}**")

                    # Show category breakdown
                    st.write("**Category Breakdown:**")
                    col1, col2, col3, col4 = st.columns(4)
                    categories = [
                        ("Standard", category_totals["standard"]),
                        ("Min Tonnage", category_totals["minimum_tonnage"]),
                        ("Overage", category_totals["overage"]),
                        ("Fees", category_totals["fees"]),
                    ]

                    for i, (cat_name, cat_total) in enumerate(categories):
                        col = [col1, col2, col3, col4][i]
                        with col:
                            if cat_total > 0:
                                st.metric(cat_name, f"${cat_total:,.2f}")
                            else:
                                st.metric(cat_name, "$0.00")

        with tab2:
            st.markdown("### Document Markdown")
            if markdown_content:
                st.markdown("**Full document markdown with detailed descriptions:**")

                # Create two sub-tabs: Rendered and Raw
                subtab1, subtab2 = st.tabs(["📖 Rendered", "📝 Raw Text"])

                with subtab1:
                    st.markdown("#### Rendered Markdown View")
                    # Clean and format the markdown for better display
                    cleaned_markdown = markdown_content.replace("\\n", "\n").replace(
                        "\\t", "\t"
                    )

                    def render_content_with_tables(content):
                        """Parse content and render tables as Streamlit dataframes"""
                        import re
                        from bs4 import BeautifulSoup
                        import pandas as pd

                        # Find all table tags
                        table_pattern = r"<table[^>]*>.*?</table>"
                        tables = re.findall(
                            table_pattern, content, re.DOTALL | re.IGNORECASE
                        )

                        if not tables:
                            # No tables, just render as markdown
                            st.markdown(content)
                            return

                        # Split content by tables to render parts separately
                        remaining_content = content
                        for i, table_html in enumerate(tables):
                            # Find the position of this table
                            parts = remaining_content.split(table_html, 1)

                            # Render content before the table
                            if parts[0].strip():
                                st.markdown(parts[0])

                            # Parse and render the table
                            try:
                                soup = BeautifulSoup(table_html, "html.parser")
                                table = soup.find("table")

                                if table:
                                    # Extract headers
                                    headers = []
                                    thead = table.find("thead")
                                    if thead:
                                        header_row = thead.find("tr")
                                        if header_row:
                                            headers = [
                                                th.get_text(strip=True).replace(
                                                    "<br>", " "
                                                )
                                                for th in header_row.find_all(
                                                    ["th", "td"]
                                                )
                                            ]

                                    # Extract rows
                                    rows = []
                                    tbody = table.find("tbody")
                                    if tbody:
                                        for row in tbody.find_all("tr"):
                                            cells = [
                                                td.get_text(strip=True)
                                                for td in row.find_all(["td", "th"])
                                            ]
                                            if cells:  # Only add non-empty rows
                                                rows.append(cells)
                                    else:
                                        # If no tbody, get all rows except header
                                        all_rows = table.find_all("tr")
                                        for row in (
                                            all_rows[1:] if headers else all_rows
                                        ):
                                            cells = [
                                                td.get_text(strip=True)
                                                for td in row.find_all(["td", "th"])
                                            ]
                                            if cells:  # Only add non-empty rows
                                                rows.append(cells)

                                    # Create and display DataFrame
                                    if rows:
                                        # Ensure all rows have the same number of columns
                                        max_cols = (
                                            max(len(row) for row in rows) if rows else 0
                                        )
                                        if headers:
                                            max_cols = max(max_cols, len(headers))

                                        # Pad rows and headers to match max columns
                                        if headers:
                                            while len(headers) < max_cols:
                                                headers.append("")
                                        else:
                                            headers = [
                                                f"Column {i+1}" for i in range(max_cols)
                                            ]

                                        for row in rows:
                                            while len(row) < max_cols:
                                                row.append("")

                                        # Create DataFrame
                                        df = pd.DataFrame(rows, columns=headers)

                                        # Display as Streamlit table with better formatting
                                        st.markdown("**Table:**")
                                        st.dataframe(
                                            df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                    else:
                                        # Fallback to markdown if table parsing fails
                                        st.markdown(f"```\n{table_html}\n```")
                                else:
                                    st.markdown(f"```\n{table_html}\n```")

                            except Exception as e:
                                st.error(f"Error parsing table: {str(e)}")
                                st.markdown(f"```\n{table_html}\n```")

                            # Update remaining content
                            remaining_content = parts[1] if len(parts) > 1 else ""

                        # Render any remaining content after the last table
                        if remaining_content.strip():
                            st.markdown(remaining_content)

                    # Split into sections for better readability
                    sections = cleaned_markdown.split("<!--")

                    for i, section in enumerate(sections):
                        if i == 0:  # First section without comment
                            render_content_with_tables(section)
                        else:
                            # Extract the comment and content
                            if "-->" in section:
                                comment_part, content_part = section.split("-->", 1)
                                # Show the source reference as an info box
                                if (
                                    "from page" in comment_part
                                    and "with ID" in comment_part
                                ):
                                    with st.expander(
                                        f"📄 Source Reference {i}", expanded=False
                                    ):
                                        st.info(f"Source: {comment_part.strip()}")
                                # Show the content
                                if content_part.strip():
                                    render_content_with_tables(content_part)
                            else:
                                render_content_with_tables(section)

                with subtab2:
                    st.markdown("#### Raw Markdown Text")
                    st.text_area(
                        "Raw Markdown Content",
                        markdown_content,
                        height=400,
                        help="This is the raw markdown representation of the document including descriptions of images, tables, and layout.",
                    )
            else:
                st.warning("No markdown content available for this document.")

        with tab3:
            st.markdown("### Raw Processing Data")

            # Show raw extraction JSON if available
            if result_data.get("extraction"):
                st.markdown("**Extracted Data (JSON):**")
                st.json(result_data["extraction"])

            # Show raw chunks data
            if result_data.get("chunks"):
                st.markdown("**Raw Chunks Data:**")
                st.json(result_data["chunks"])

            if not result_data.get("extraction") and not result_data.get("chunks"):
                st.warning("No raw processing data available")

        with tab4:
            st.markdown("### Processing Summary")
            st.write(f"**File:** {filename}")
            st.write(f"**Processed:** {metadata.get('processed_at', 'Unknown')}")
            st.write(f"**File Hash:** {metadata.get('file_hash', 'Unknown')[:16]}...")

            if result_data.get("extraction"):
                extraction = result_data["extraction"]
                st.write(
                    f"**Invoice Number:** {extraction.get('invoice_number', 'N/A')}"
                )
                st.write(f"**Customer:** {extraction.get('customer_name', 'N/A')}")
                st.write(
                    f"**Total Amount:** ${extraction.get('current_invoice_charges', 'N/A')}"
                )
    else:
        # For invoices without structured extraction, show basic tabs including markdown
        tab1, tab2, tab3 = st.tabs(["📄 Markdown", "🔧 Raw JSON", "📋 Summary"])

        with tab1:
            st.markdown("### Document Markdown")
            if markdown_content:
                st.markdown("**Full document markdown with detailed descriptions:**")

                # Create two sub-tabs: Rendered and Raw
                subtab1, subtab2 = st.tabs(["📖 Rendered", "📝 Raw Text"])

                with subtab1:
                    st.markdown("#### Rendered Markdown View")
                    # Clean and format the markdown for better display
                    cleaned_markdown = markdown_content.replace("\\n", "\n").replace(
                        "\\t", "\t"
                    )

                    def render_content_with_tables(content):
                        """Parse content and render tables as Streamlit dataframes"""
                        import re
                        from bs4 import BeautifulSoup
                        import pandas as pd

                        # Find all table tags
                        table_pattern = r"<table[^>]*>.*?</table>"
                        tables = re.findall(
                            table_pattern, content, re.DOTALL | re.IGNORECASE
                        )

                        if not tables:
                            # No tables, just render as markdown
                            st.markdown(content)
                            return

                        # Split content by tables to render parts separately
                        remaining_content = content
                        for i, table_html in enumerate(tables):
                            # Find the position of this table
                            parts = remaining_content.split(table_html, 1)

                            # Render content before the table
                            if parts[0].strip():
                                st.markdown(parts[0])

                            # Parse and render the table
                            try:
                                soup = BeautifulSoup(table_html, "html.parser")
                                table = soup.find("table")

                                if table:
                                    # Extract headers
                                    headers = []
                                    thead = table.find("thead")
                                    if thead:
                                        header_row = thead.find("tr")
                                        if header_row:
                                            headers = [
                                                th.get_text(strip=True).replace(
                                                    "<br>", " "
                                                )
                                                for th in header_row.find_all(
                                                    ["th", "td"]
                                                )
                                            ]

                                    # Extract rows
                                    rows = []
                                    tbody = table.find("tbody")
                                    if tbody:
                                        for row in tbody.find_all("tr"):
                                            cells = [
                                                td.get_text(strip=True)
                                                for td in row.find_all(["td", "th"])
                                            ]
                                            if cells:  # Only add non-empty rows
                                                rows.append(cells)
                                    else:
                                        # If no tbody, get all rows except header
                                        all_rows = table.find_all("tr")
                                        for row in (
                                            all_rows[1:] if headers else all_rows
                                        ):
                                            cells = [
                                                td.get_text(strip=True)
                                                for td in row.find_all(["td", "th"])
                                            ]
                                            if cells:  # Only add non-empty rows
                                                rows.append(cells)

                                    # Create and display DataFrame
                                    if rows:
                                        # Ensure all rows have the same number of columns
                                        max_cols = (
                                            max(len(row) for row in rows) if rows else 0
                                        )
                                        if headers:
                                            max_cols = max(max_cols, len(headers))

                                        # Pad rows and headers to match max columns
                                        if headers:
                                            while len(headers) < max_cols:
                                                headers.append("")
                                        else:
                                            headers = [
                                                f"Column {i+1}" for i in range(max_cols)
                                            ]

                                        for row in rows:
                                            while len(row) < max_cols:
                                                row.append("")

                                        # Create DataFrame
                                        df = pd.DataFrame(rows, columns=headers)

                                        # Display as Streamlit table with better formatting
                                        st.markdown("**Table:**")
                                        st.dataframe(
                                            df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                    else:
                                        # Fallback to markdown if table parsing fails
                                        st.markdown(f"```\n{table_html}\n```")
                                else:
                                    st.markdown(f"```\n{table_html}\n```")

                            except Exception as e:
                                st.error(f"Error parsing table: {str(e)}")
                                st.markdown(f"```\n{table_html}\n```")

                            # Update remaining content
                            remaining_content = parts[1] if len(parts) > 1 else ""

                        # Render any remaining content after the last table
                        if remaining_content.strip():
                            st.markdown(remaining_content)

                    # Split into sections for better readability
                    sections = cleaned_markdown.split("<!--")

                    for i, section in enumerate(sections):
                        if i == 0:  # First section without comment
                            render_content_with_tables(section)
                        else:
                            # Extract the comment and content
                            if "-->" in section:
                                comment_part, content_part = section.split("-->", 1)
                                # Show the source reference as an info box
                                if (
                                    "from page" in comment_part
                                    and "with ID" in comment_part
                                ):
                                    with st.expander(
                                        f"📄 Source Reference {i}", expanded=False
                                    ):
                                        st.info(f"Source: {comment_part.strip()}")
                                # Show the content
                                if content_part.strip():
                                    render_content_with_tables(content_part)
                            else:
                                render_content_with_tables(section)

                with subtab2:
                    st.markdown("#### Raw Markdown Text")
                    st.text_area(
                        "Raw Markdown Content",
                        markdown_content,
                        height=400,
                        help="This is the raw markdown representation of the document including descriptions of images, tables, and layout.",
                    )
            else:
                st.warning("No markdown content available for this document.")

        with tab2:
            st.markdown("### Raw Processing Data")
            if result_data.get("chunks"):
                st.json(result_data["chunks"])
            else:
                st.warning("No raw processing data available")

        with tab3:
            st.markdown("### Processing Summary")
            st.write(f"**File:** {filename}")
            st.write(f"**Processed:** {metadata.get('processed_at', 'Unknown')}")
            st.write(f"**File Hash:** {metadata.get('file_hash', 'Unknown')[:16]}...")


def invoice_summary_page():
    """Page showing a compact table of all processed invoices with filters"""
    st.header("📊 Invoice Summary Table")

    # Load all results
    results = load_all_results()

    if not results:
        st.info("No invoices processed yet. Process some invoices first!")
        return

    # Extract invoice data for table
    table_data = []
    for result in results:
        try:
            metadata = result["metadata"]
            data = result["data"]
            extraction = data.get("extraction", {})

            if extraction:  # Only include results with structured extraction
                # Parse date strings safely
                invoice_date = extraction.get("invoice_date", "")
                processed_at = metadata.get("processed_at", "")

                # Count line items
                line_items = extraction.get("line_items", [])
                num_line_items = len(line_items) if isinstance(line_items, list) else 0

                # Parse invoice total safely
                invoice_total = extraction.get("current_invoice_charges")
                if isinstance(invoice_total, str):
                    try:
                        invoice_total = float(
                            invoice_total.replace("$", "").replace(",", "")
                        )
                    except:
                        invoice_total = 0.0
                elif invoice_total is None:
                    invoice_total = 0.0

                # Calculate category breakdowns and tonnage
                category_totals = {
                    "standard": 0,
                    "minimum_tonnage": 0,
                    "overage": 0,
                    "fees": 0,
                }
                total_tons = 0

                for item_data in line_items:
                    try:
                        line_item = LineItem(**item_data)
                        category = line_item.category.value
                        category_totals[category] += item_data.get("amount", 0)
                        total_tons += line_item.estimated_tons
                    except Exception:
                        continue

                table_data.append(
                    {
                        "Customer Number": extraction.get("customer_id", "N/A"),
                        "Invoice Number": extraction.get("invoice_number", "N/A"),
                        "Invoice Date": invoice_date,
                        "Line Items": num_line_items,
                        "Total Tons": total_tons,
                        "Standard ($)": category_totals["standard"],
                        "Min Tonnage ($)": category_totals["minimum_tonnage"],
                        "Overage ($)": category_totals["overage"],
                        "Fees ($)": category_totals["fees"],
                        "Invoice Total": invoice_total,
                        "Processed Date": processed_at,
                        "_filename": metadata.get(
                            "filename", "N/A"
                        ),  # Keep for selection but don't display
                    }
                )
        except Exception as e:
            continue

    if not table_data:
        st.warning("No invoices with extracted data found.")
        return

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Create top layout with equal height sections
    st.markdown("---")
    top_col1, top_col2 = st.columns([1, 2])

    with top_col1:
        with st.container():
            st.markdown("### 🔍 Filters")

            # Customer filter
            unique_customers = ["All"] + sorted(df["Customer Number"].unique().tolist())
            customer_filter = st.selectbox(
                "Customer", unique_customers, key="customer_filter"
            )

            # Invoice total threshold
            min_total = float(df["Invoice Total"].min()) if not df.empty else 0.0
            max_total = float(df["Invoice Total"].max()) if not df.empty else 10000.0
            total_threshold = st.number_input(
                "Min Total ($)",
                min_value=0.0,
                max_value=max_total,
                value=0.0,
                step=100.0,
                key="total_threshold",
            )

            # Date range filter
            st.markdown("**Date Range**")
            # Convert invoice dates to datetime for filtering
            df_dates = df[df["Invoice Date"] != ""].copy()
            if not df_dates.empty:
                try:
                    df_dates["Invoice Date Parsed"] = pd.to_datetime(
                        df_dates["Invoice Date"], errors="coerce"
                    )
                    df_dates = df_dates.dropna(subset=["Invoice Date Parsed"])

                    if not df_dates.empty:
                        min_date = df_dates["Invoice Date Parsed"].min().date()
                        max_date = df_dates["Invoice Date Parsed"].max().date()

                        date_from = st.date_input(
                            "From",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="date_from",
                        )
                        date_to = st.date_input(
                            "To",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="date_to",
                        )
                    else:
                        date_from = date_to = None
                except:
                    date_from = date_to = None
            else:
                date_from = date_to = None

    # Apply filters
    filtered_df = df.copy()

    if customer_filter != "All":
        filtered_df = filtered_df[filtered_df["Customer Number"] == customer_filter]

    if total_threshold > 0:
        filtered_df = filtered_df[filtered_df["Invoice Total"] >= total_threshold]

    # Date range filter
    if date_from and date_to:
        try:
            filtered_df["Invoice Date Parsed"] = pd.to_datetime(
                filtered_df["Invoice Date"], errors="coerce"
            )
            filtered_df = filtered_df.dropna(subset=["Invoice Date Parsed"])
            filtered_df = filtered_df[
                (filtered_df["Invoice Date Parsed"].dt.date >= date_from)
                & (filtered_df["Invoice Date Parsed"].dt.date <= date_to)
            ]
            filtered_df = filtered_df.drop(columns=["Invoice Date Parsed"])
        except:
            pass

    with top_col2:
        with st.container():
            st.markdown("### 📈 Statistics")

            if not filtered_df.empty:
                # Summary stats in a compact layout matching filter height
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    st.metric("Invoices", len(filtered_df))
                    min_amount = filtered_df["Invoice Total"].min()
                    st.metric("Min Amount", f"${min_amount:,.0f}")

                with metric_col2:
                    total_amount = filtered_df["Invoice Total"].sum()
                    st.metric("Total", f"${total_amount:,.0f}")
                    max_amount = filtered_df["Invoice Total"].max()
                    st.metric("Max Amount", f"${max_amount:,.0f}")

                with metric_col3:
                    avg_amount = filtered_df["Invoice Total"].mean()
                    st.metric("Average", f"${avg_amount:,.0f}")
                    unique_customers = filtered_df["Customer Number"].nunique()
                    st.metric("Customers", unique_customers)

                with metric_col4:
                    avg_line_items = filtered_df["Line Items"].mean()
                    st.metric("Avg Line Items", f"{avg_line_items:.1f}")
                    total_tons = filtered_df["Total Tons"].sum()
                    st.metric("Total Tons", f"{total_tons:,.1f}")
            else:
                st.info("No data to display - adjust filters to see statistics")

    # Full width table section below
    st.markdown("---")
    # Display filtered table - full width
    st.markdown(f"### 📋 Invoice Table ({len(filtered_df)} invoices)")

    if not filtered_df.empty:
        # Format the display DataFrame
        display_df = filtered_df.copy()

        # Format currency columns
        currency_columns = [
            "Standard ($)",
            "Min Tonnage ($)",
            "Overage ($)",
            "Fees ($)",
            "Invoice Total",
        ]
        for col in currency_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) and x != 0 else "$0.00"
                )

        # Format tons column
        if "Total Tons" in display_df.columns:
            display_df["Total Tons"] = display_df["Total Tons"].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) and x != 0 else "0.00"
            )

        # Reorder columns for better display
        column_order = [
            "Customer Number",
            "Invoice Number",
            "Invoice Date",
            "Line Items",
            "Total Tons",
            "Standard ($)",
            "Min Tonnage ($)",
            "Overage ($)",
            "Fees ($)",
            "Invoice Total",
            "Processed Date",
        ]
        display_df = display_df[column_order]

        # Display with selection capability
        selected_indices = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Option to view details of selected invoice
        if hasattr(selected_indices, "selection") and selected_indices.selection.rows:
            selected_idx = selected_indices.selection.rows[0]
            selected_filename = filtered_df.iloc[selected_idx]["_filename"]

            # Find the full result data for the selected invoice
            for result in results:
                if result["metadata"].get("filename") == selected_filename:
                    display_single_result(result["data"], selected_filename)
                    break
    else:
        st.info("No invoices match the current filters.")


if __name__ == "__main__":
    main()
