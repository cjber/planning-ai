import py7zr
import streamlit as st

from planning_ai.common.utils import Paths
from planning_ai.main import main as report_main
from planning_ai.preprocessing.azure_doc import azure_process_pdfs
from planning_ai.preprocessing.gcpt3 import main as preprocess_main

UPLOAD_DIR = Paths.RAW / "gcpt3"

if "files_extracted" not in st.session_state:
    st.session_state["files_extracted"] = False
if "files_processed" not in st.session_state:
    st.session_state["files_processed"] = False
if "pdfs_processed" not in st.session_state:
    st.session_state["pdfs_processed"] = False

st.title("Planning AI")


st.header("1. Upload JDL response `.json` files")
st.write(
    "Upload your `.json` files here as a `7zip` file, they will be saved to the `data/raw/gcpt3` directory."
)
uploaded_file = st.file_uploader("Choose a `.7z` file.", type="7z")

if uploaded_file and not st.session_state["files_extracted"]:
    with st.spinner("Extracting files..."):
        try:
            with py7zr.SevenZipFile(uploaded_file, mode="r") as archive:
                archive.extractall(path=UPLOAD_DIR)
            st.session_state["files_extracted"] = True
            st.write(f"Extracted all files to `{UPLOAD_DIR}`.")
        except Exception as e:
            st.error(f"Failed to extract files {e}")

if not st.session_state["files_extracted"]:
    st.write("No files uploaded yet.")

if st.session_state["files_extracted"]:
    st.header("2. Process uploaded `.json` files")
    st.write(
        "Once the files are extracted, click the button below to start preprocessing the `.json` files."
    )
    if st.button("Process Files"):
        with st.spinner("Running preprocessing..."):
            try:
                preprocess_main()
                st.session_state["files_processed"] = True
                st.success("Preprocessing completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during preprocessing: {e}")

if st.session_state["files_extracted"] and st.session_state["files_processed"]:
    st.header("3. Extract text from PDFs.")
    st.write(
        "After preprocessing the `.json` files, you can now extract text from the PDFs by clicking the button below."
    )
    if st.button("Process PDFs"):
        with st.spinner("Extracting text from PDFs..."):
            try:
                azure_process_pdfs()
                st.session_state["pdfs_processed"] = True
                st.success("Text extraction completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during PDF text extraction: {e}")

if (
    st.session_state["files_extracted"]
    and st.session_state["files_processed"]
    and st.session_state["pdfs_processed"]
):
    st.title("Build final report.")
    st.write(
        "After extracting text from PDFs, you can now run the full report building pipeline!"
    )
    if st.button("Build Report", type="primary"):
        with st.spinner("Building report..."):
            try:
                report_main()
            except Exception as e:
                st.error(f"An error occurred during report building: {e}")
            report_path = Paths.SUMMARY / "Summary_Documents.pdf"
            summaries_path = Paths.SUMMARY / "Summary_of_Submitted_Responses.pdf"

            if report_path.exists() and summaries_path.exists():
                st.success("Report built successfully! Check the `data/out` directory.")
                col1, col2 = st.columns(2)
                with col1:
                    with open(summaries_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download Report",
                            data=pdf_file,
                            file_name="Summary_of_Submitted_Responses.pdf",
                            mime="application/pdf",
                        )
                with col2:
                    with open(report_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download Summaries Report",
                            data=pdf_file,
                            file_name="Summary_Documents.pdf",
                            mime="application/pdf",
                        )
