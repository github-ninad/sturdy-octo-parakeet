import shutil
import time
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

from agents.adjudication_workflow import process_claim_detailed, get_detailed_adjudication_prompt


def set_custom_style():
    """Set custom CSS styles"""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding: 2rem;
        }

        /* Custom header styling */
        .custom-header {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }

        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #1f77b4;
            border-radius: 5px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #1f77b4;
        }

        /* Custom card styling */
        .custom-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }

        /* Success message styling */
        .success-message {
            color: #28a745;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }

        /* Warning message styling */
        .warning-message {
            color: #ffc107;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def create_custom_header():
    """Create a custom header with description"""
    st.markdown("""
        <div class='custom-header'>
            <h1>Multi-Agent Claim Adjudication</h1>
            <p style='font-size: 1.1em; color: #666;'>
                Upload your medical documents for intelligent processing and analysis
            </p>
        </div>
    """, unsafe_allow_html=True)


def create_file_upload_section():
    """Create an enhanced file upload section"""
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("📄 Document Upload")
    st.markdown("Please upload the required documents in PDF format.")

    col1, col2 = st.columns(2)

    with col1:
        claim_file = st.file_uploader(
            "Claim Form (Required)",
            type=["pdf"],
            key="claim_form",
            help="Upload the official claim form document"
        )

        discharge_file = st.file_uploader(
            "Discharge Note (Required)",
            type=["pdf"],
            key="discharge_note",
            help="Upload the hospital discharge summary"
        )

    with col2:
        bills_file = st.file_uploader(
            "Medical Bills (Required)",
            type=["pdf"],
            key="medical_bills",
            help="Upload all medical bills and invoices"
        )

        other_files = st.file_uploader(
            "Supporting Documents (Optional)",
            type=["pdf"],
            accept_multiple_files=True,
            key="other_docs",
            help="Upload any additional supporting documents"
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return claim_file, discharge_file, bills_file, other_files


def show_document_status(uploaded_files):
    """Display status of uploaded documents"""
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("📋 Document Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Required Documents")
        st.markdown(f"- Claim Form: {'✅' if uploaded_files[0] else '⌛'}")
        st.markdown(f"- Discharge Note: {'✅' if uploaded_files[1] else '⌛'}")
        st.markdown(f"- Medical Bills: {'✅' if uploaded_files[2] else '⌛'}")

    with col2:
        st.markdown("### Additional Documents")
        if uploaded_files[3]:
            for doc in uploaded_files[3]:
                st.markdown(f"- ✅ {doc.name}")
        else:
            st.markdown("- No additional documents uploaded")

    st.markdown("</div>", unsafe_allow_html=True)


def show_processing_progress():
    """Show processing progress with custom styling"""
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Processing Status")

    progress_bar = st.progress(0)
    status_text = st.empty()

    stages = [
        "Initializing document processor...",
        "Extracting document content...",
        "Analyzing medical information...",
        "Validating claim details...",
        "Generating final report..."
    ]

    for i, stage in enumerate(stages):
        progress = (i + 1) * 20
        status_text.text(stage)
        progress_bar.progress(progress)
        time.sleep(0.5)

    st.markdown("</div>", unsafe_allow_html=True)


def show_results(results):
    """Display processing results with enhanced styling"""
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("📊 Processing Results")

    tabs = st.tabs(["Summary", "Detailed View", "Export Options"])

    with tabs[0]:
        st.markdown("### Quick Summary")
        for result in results:
            if result["status"] == "success":
                st.success(f"✅ Successfully processed {result['file_name']}")
            else:
                st.error(f"❌ Error processing {result['file_name']}")

    with tabs[1]:
        for result in results:
            with st.expander(f"📄 {result['file_name']} Details"):
                if result["status"] == "success":
                    st.markdown(result["content"], unsafe_allow_html=True)
                else:
                    st.error(f"Error: {result['error']}")

    with tabs[2]:
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Full Report (PDF)",
                data="report_data",
                file_name="claim_report.pdf",
                mime="application/pdf"
            )
        with col2:
            st.download_button(
                "📥 Download Summary (Excel)",
                data="summary_data",
                file_name="claim_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.markdown("</div>", unsafe_allow_html=True)

# Configure page
st.set_page_config(
    page_title="Health Claims Document Processor",
    layout="wide"
)


def setup_temp_directory():
    """Create temporary directory for file storage"""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def save_uploaded_file(uploaded_file, temp_dir: Path) -> Path:
    """Save uploaded file to temporary directory and return path"""
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path



def process_document(file_path: Path, doc_type) -> Dict[str, Any]:
    """Process document using Docling"""
    try:
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = doc_converter.convert(str(file_path))
        return {
            "status": "success",
            "content": result.document.export_to_markdown(),
            "file_name": file_path.name,
            "doc_type": doc_type
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "file_name": file_path.name,
            "doc_type": doc_type
        }


def process_documents(files: list, temp_dir: Path) -> List[Dict[str, Any]]:
    """
    Process multiple documents with improved error handling and status tracking

    Args:
        files: List of [claim_file, discharge_file, bills_file, other_files]
        temp_dir: Path to temporary directory

    Returns:
        List of processing results for each document
    """
    results = []

    # Process main documents (claim, discharge, bills)
    doc_types = ['Claim Form', 'Discharge Note', 'Medical Bills']
    for file_obj, doc_type in zip(files[:3], doc_types):
        if file_obj is not None:
            try:
                file_path = save_uploaded_file(file_obj, temp_dir)
                result = process_document(file_path, doc_type)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e),
                    "file_name": file_obj.name,
                    "doc_type": doc_type
                })

    # Process additional documents if any
    other_files = files[3] or []
    for file_obj in other_files:
        try:
            file_path = save_uploaded_file(file_obj, temp_dir)
            result = process_document(file_path, 'Supporting Document')
            results.append(result)
        except Exception as e:
            results.append({
                "status": "error",
                "error": str(e),
                "file_name": file_obj.name,
                "doc_type": 'Supporting Document'
            })

    return results


def prepare_content(results: List[Dict[str, Any]]) -> str:
    """
    Prepare processed document content for adjudication

    Args:
        results: List of document processing results

    Returns:
        Formatted string containing all document content
    """
    content = ""

    # Sort results by document type to ensure proper order
    doc_type_order = {
        'Claim Form': 1,
        'Discharge Note': 2,
        'Medical Bills': 3,
        'Supporting Document': 4
    }

    sorted_results = sorted(
        results,
        key=lambda x: doc_type_order.get(x['doc_type'], 999)
    )

    for result in sorted_results:
        if result["status"] == "success":
            content += f"""
{"=" * 50}

Document Type: {result["doc_type"]}
File Name: {result["file_name"]}

Content:
{result["content"]}

"""
        else:
            content += f"""
{"=" * 50}

Document Type: {result["doc_type"]}
File Name: {result["file_name"]}
Status: Error processing document
Error Details: {result["error"]}

"""

    return content


def cleanup_temp_files(temp_dir: Path):
    """
    Clean up temporary files and directory

    Args:
        temp_dir: Path to temporary directory
    """
    try:
        if temp_dir.exists():
            # Remove all files in the directory
            for file_path in temp_dir.glob("*"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                except Exception as e:
                    st.warning(f"Failed to remove file {file_path}: {str(e)}")

            # Remove the directory itself
            shutil.rmtree(temp_dir)

    except Exception as e:
        st.warning(f"Failed to clean up temporary directory: {str(e)}")
        # If rmtree fails, try to warn user but don't crash
        pass


def format_chunk_to_markdown(chunk_content) -> str:
    """
    Convert a chunk to markdown only if it's a dictionary/object,
    otherwise return the text as is

    Args:
        chunk_content: Either string or dictionary with specific keys

    Returns:
        Formatted markdown string or original text
    """
    # If it's not a dictionary, return the content as is
    if not isinstance(chunk_content, dict):
        return str(chunk_content)

    try:
        # For dictionary content, create a structured box with sections
        sections = {
            "📋 Title": chunk_content.get('title', ''),
            "🎯 Action": chunk_content.get('action', ''),
            "📊 Result": chunk_content.get('result', ''),
            "💭 Reasoning": chunk_content.get('reasoning', ''),
            "📈 Confidence": f"{float(chunk_content.get('confidence', 0)) * 100:.0f}%"
        }

        # Build markdown with styled sections
        markdown = """<div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin:10px 0; background-color:#f8f9fa;">"""

        for label, content in sections.items():
            if content:  # Only include non-empty sections
                markdown += f"""
<div style="margin-bottom:10px;">
<strong>{label}:</strong><br/>{content}
</div>"""

        markdown += "</div>"
        return markdown

    except Exception as e:
        # If any error in formatting the dictionary, return the raw content
        return str(chunk_content)


def main():
    set_custom_style()
    create_custom_header()

    # Setup temporary directory
    temp_dir = setup_temp_directory()

    # File upload section
    claim_file, discharge_file, bills_file, other_files = create_file_upload_section()

    # Show document status
    show_document_status([claim_file, discharge_file, bills_file, other_files])

    # Process button with enhanced styling
    if st.button("🚀 Process Documents", use_container_width=True):
        if any([claim_file, discharge_file, bills_file, other_files]):
            show_processing_progress()

            results = process_documents(
                [claim_file, discharge_file, bills_file, other_files],
                temp_dir
            )

            st.session_state.processed_results = results
            show_results(results)

        else:
            st.warning("⚠️ Please upload at least one document to process")

    # Show final report
    if st.session_state.get("processed_results"):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📑 Final Analysis")

        text_placeholder = st.empty()

        with st.spinner("🤖 AI Agents analyzing your claim..."):
            response_stream = process_claim_detailed(
                prepare_content(st.session_state.processed_results)
            )
            # Modified streaming code
            response_collector = ""
            for chunk in response_stream:
                try:
                    # Get the content from the chunk
                    if isinstance(chunk, dict):
                        chunk_res = chunk
                    else:
                        chunk_res = chunk.model_dump(exclude={"messages"})

                    # Extract content (could be string or dict)
                    content = chunk_res.get('content', chunk_res)

                    # Format the content
                    formatted_markdown = format_chunk_to_markdown(content)

                    # Add to collector
                    response_collector += formatted_markdown

                    # Update display
                    text_placeholder.markdown(response_collector, unsafe_allow_html=True)

                except Exception as e:
                    # Handle any errors gracefully
                    error_box = f"""
            <div style="border:1px solid #ffcdd2; border-radius:5px; padding:10px; margin:10px 0; background-color:#ffebee;">
            Error processing chunk: {str(e)}
            </div>
            """
                    response_collector += error_box
                    text_placeholder.markdown(response_collector, unsafe_allow_html=True)

            # text_placeholder.markdown(response_stream.content, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Cleanup on session end
    if st.session_state.get("_is_running", False):
        cleanup_temp_files(temp_dir)


if __name__ == "__main__":
    main()
