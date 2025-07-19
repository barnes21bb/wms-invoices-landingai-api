import streamlit as st
import os
import tempfile
import json
from dotenv import load_dotenv
from agentic_doc.parse import parse

# Load environment variables
load_dotenv()


def main():
    st.set_page_config(
        page_title="LandingAI Document Processor", page_icon="📄", layout="wide"
    )

    st.title("📄 LandingAI Document Processor")
    st.markdown("Upload documents and extract structured data using LandingAI's API")

    # Check API key
    api_key = os.getenv("VISION_AGENT_API_KEY")
    if not api_key:
        st.error("❌ VISION_AGENT_API_KEY not found in environment variables")
        st.info(
            "Please set your API key in the .env file or as an environment variable"
        )
        return

    st.success("✅ API Key loaded successfully")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document to process",
        type=["pdf", "png", "jpg", "jpeg", "doc", "docx"],
        help="Upload PDF, image, or document files",
    )

    if uploaded_file is not None:
        # Display file info
        st.subheader("📋 File Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
            st.write(f"**File type:** {uploaded_file.type}")

        # Process button
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document with LandingAI..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f"_{uploaded_file.name}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    # Process with LandingAI
                    result = parse(tmp_file_path)

                    # Clean up temp file
                    os.unlink(tmp_file_path)

                    # Store results in session state
                    st.session_state.processing_result = result
                    st.session_state.processed_file = uploaded_file.name

                    st.success("✅ Document processed successfully!")

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    return

    # Display results if available
    if (
        hasattr(st.session_state, "processing_result")
        and st.session_state.processing_result
    ):
        st.subheader("📊 Processing Results")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📝 Markdown", "🔧 JSON Chunks", "📋 Summary"])

        with tab1:
            st.markdown("### Extracted Markdown")
            if hasattr(st.session_state.processing_result, "markdown"):
                st.markdown(st.session_state.processing_result.markdown)
            else:
                st.warning("No markdown content available")

        with tab2:
            st.markdown("### Structured Data (JSON)")
            if hasattr(st.session_state.processing_result, "chunks"):
                st.json(st.session_state.processing_result.chunks)
            else:
                st.warning("No structured data available")

        with tab3:
            st.markdown("### Processing Summary")
            result = st.session_state.processing_result

            # Display metadata
            st.write(f"**Processed file:** {st.session_state.processed_file}")

            # Show available attributes
            st.write("**Available result attributes:**")
            for attr in dir(result):
                if not attr.startswith("_"):
                    st.write(f"- {attr}")

        # Download buttons
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)

        with col1:
            if hasattr(st.session_state.processing_result, "markdown"):
                st.download_button(
                    label="📄 Download Markdown",
                    data=st.session_state.processing_result.markdown,
                    file_name=f"{st.session_state.processed_file}_extracted.md",
                    mime="text/markdown",
                )

        with col2:
            if hasattr(st.session_state.processing_result, "chunks"):
                json_data = json.dumps(
                    st.session_state.processing_result.chunks, indent=2
                )
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"{st.session_state.processed_file}_extracted.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
