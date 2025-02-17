import os
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError
from tqdm import tqdm

from planning_ai.common.utils import Paths

load_dotenv()

endpoint = os.getenv("AZURE_API_ENDPOINT") or ""
credential = AzureKeyCredential(os.getenv("AZURE_API_KEY") or "")
document_intelligence_client = DocumentIntelligenceClient(endpoint, credential)


def read_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n\n".join([page.extract_text() for page in reader.pages])
        return text, reader
    except PdfReadError:
        print("Not a pdf file...")
        return None, None


def write_pdf(reader, out_pdf):
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    with open(out_pdf, "wb") as f:
        writer.write(f)
    print("Written PDF text to file.")


def analyze_document_with_azure(pdf_path, out_pdf, failed_txt):
    with open(pdf_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-read",
            body=f,
            output=[AnalyzeOutputOption.PDF],
        )
    try:
        result: AnalyzeResult = poller.result()
        operation_id = poller.details["operation_id"]

        response = document_intelligence_client.get_analyze_result_pdf(
            model_id=result.model_id, result_id=operation_id
        )
        with open(out_pdf, "wb") as writer:
            writer.writelines(response)
        print("Written Azure text to file.")
    except Exception as e:
        with open(failed_txt, "w") as f:
            f.write("")
        print(f"Error occurred in result. {e}")


def azure_process_pdfs():
    pdfs = (Paths.RAW / "pdfs").glob("*.pdf")

    for pdf_path in tqdm(pdfs):
        print(f"Processing {pdf_path}")

        out_pdf = Path(f"./data/staging/pdfs_azure/{pdf_path.stem}.pdf")
        failed_txt = Path(f"./data/staging/pdfs_azure/{pdf_path.stem}.txt")

        text, reader = read_pdf(pdf_path)
        if text is None:
            with open(failed_txt, "w") as f:
                f.write("")
            continue

        if len(text) > 10_000:
            write_pdf(reader, out_pdf)

        if out_pdf.exists() or failed_txt.exists():
            continue

        if pdf_path.stat().st_size > 1_000_000:
            with open(failed_txt, "w") as f:
                f.write("")
            print("PDF too large!")
            continue

        analyze_document_with_azure(pdf_path, out_pdf, failed_txt)


if __name__ == "__main__":
    azure_process_pdfs()
