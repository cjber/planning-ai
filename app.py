import polars as pl
import py7zr
import streamlit as st
import streamlit_authenticator as stauth

from planning_ai.common.utils import Paths
from planning_ai.main import main as report_main
from planning_ai.preprocessing.azure_doc import azure_process_pdfs
from planning_ai.preprocessing.gcpt3 import main as preprocess_main

auth = st.secrets.to_dict()

authenticator = stauth.Authenticate(
    auth["credentials"],
    auth["cookie"]["name"],
    auth["cookie"]["key"],
    auth["cookie"]["expiry_days"],
)

UPLOAD_DIR = Paths.RAW / "gcpt3"

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if "files_extracted" not in st.session_state:
    st.session_state["files_extracted"] = False
if "completed" not in st.session_state:
    st.session_state["completed"] = False

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write("---")

    st.title("Report Builder")

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
                # remove old files
                _ = [file.unlink() for file in UPLOAD_DIR.glob("*.json")]

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

    if st.session_state["files_extracted"] and not st.session_state["completed"]:
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
                representations_documents = report_main()
                st.session_state["completed"] = True
elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] is None:
    st.warning("Please enter your username and password")

if st.session_state["completed"]:
    representations_documents = (
        pl.read_parquet(Paths.STAGING / "gcpt3.parquet")["representations_document"]
        .unique()
        .to_list()
    )

    st.success("Reports built successfully! Please click download buttons below.")
    for rep in representations_documents:
        report_path = Paths.SUMMARY / f"Summary_Documents-{rep}.pdf"
        summaries_path = Paths.SUMMARY / f"Summary_of_Submitted_Responses-{rep}.pdf"

        col1, col2 = st.columns(2, border=True)
        with col1:
            with open(summaries_path, "rb") as pdf_file:
                st.markdown("**Representations Summary Download**")
                st.download_button(
                    label=f"{rep}",
                    data=pdf_file,
                    file_name=f"Summary_of_Submitted_Responses-{rep}.pdf",
                    mime="application/pdf",
                    type="primary",
                )
        with col2:
            with open(report_path, "rb") as pdf_file:
                st.markdown("**Executive Report Download**")
                st.download_button(
                    label=f"{rep}",
                    data=pdf_file,
                    file_name=f"Summary_Documents-{rep}.pdf",
                    mime="application/pdf",
                    type="primary",
                )
