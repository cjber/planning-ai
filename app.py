import py7zr
import streamlit as st

from planning_ai.common.utils import Paths
from planning_ai.main import main as report_main
from planning_ai.preprocessing.azure_doc import azure_process_pdfs
from planning_ai.preprocessing.gcpt3 import main as preprocess_main

UPLOAD_DIR = Paths.RAW / "gcpt3"

if "files_extracted" not in st.session_state:
    st.session_state["files_extracted"] = False

st.title("Planning AI")


st.header("Upload JDL response `.json` files")
st.write(
    "Upload your `.json` files here as a `7zip` file, they will be saved to the `data/raw/gcpt3` directory."
)

with st.expander("File Format"):
    st.write(
        """
    The `.json` files should look like the following:

    ```json
    {
        "id": 10008,
        "method": "Paper",
        "respondentpostcode": "CB2 9NE",
        "text": "",
        "attachments": [
            {
                "id": 3803,
                "url": "http:\/\/www.cambridge.gov.uk\/public\/ldf\/localplan2031\/15417.pdf",
                "published": false
            }
        ],
        "representations": [
            {
                "id": 15417,
                "support\/object": "Object",
                "document": "Issues and Options Report",
                "documentelementid": 29785,
                "documentelementtitle": "3 - Spatial Strategy, Question 3.10",
                "summary": "No more green belt taken away, which is prime agricultural land. Noise pollution & light pollution for surrounding villages and new houses being built, no bus services either!"
            },
        ]
    }
    ```
"""
    )
if uploaded_file := st.file_uploader("Choose a `.7z` file:", type="7z"):
    with st.spinner("Extracting files..."):
        try:
            with py7zr.SevenZipFile(uploaded_file, mode="r") as archive:
                archive.extractall(path=UPLOAD_DIR)
            st.session_state["files_extracted"] = True
            st.success(
                f"Extracted `{len(list(UPLOAD_DIR.glob('*.json')))}` files to `{UPLOAD_DIR}`."
            )
        except Exception as e:
            st.error(f"Failed to extract files {e}")

if not st.session_state["files_extracted"]:
    st.write("No files uploaded yet.")

st.write("---")

if st.session_state["files_extracted"]:
    st.title("Build Report")
    st.write(
        "Once the files are extracted, click the button below to build the report."
    )
    if st.button("Build Report", type="primary"):
        with st.spinner("Preprocessing files..."):
            try:
                preprocess_main()
                st.success("Preprocessing completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during preprocessing: {e}")
        with st.spinner("Extracting text from PDFs..."):
            try:
                azure_process_pdfs()
                st.success("Text extraction completed successfully!")
            except Exception as e:
                st.error(f"An error occurred during PDF text extraction: {e}")
        with st.spinner("Building report..."):
            report_main()
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
