import os
from pathlib import Path

import polars as pl
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from tqdm import tqdm

from planning_ai.common.utils import Paths

gcpt3 = pl.read_parquet("data/staging/gcpt3_testing.parquet")
pdf_ids = gcpt3["attachments_id"].unique().to_list()

pdfs = [
    pdf
    for pdf in (Paths.RAW / "pdfs").glob("*.pdf")
    if pdf.stem.isdigit() and int(pdf.stem) in pdf_ids
]


endpoint = os.getenv("AZURE_API_ENDPOINT") or ""
credential = AzureKeyCredential(os.getenv("AZURE_API_KEY") or "")
document_intelligence_client = DocumentIntelligenceClient(endpoint, credential)

for pdf_path in tqdm(pdfs):
    out_pdf = Path(f"./data/staging/pdfs_azure/{pdf_path.stem}.pdf")
    failed_txt = Path(f"./data/staging/pdfs_azure/{pdf_path.stem}.txt")
    if out_pdf.exists() or failed_txt.exists():
        continue

    if pdf_path.stat().st_size > 1_000_000:
        with open(failed_txt, "w") as f:
            f.write("")
        print("PDF too large!")
        continue

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
    except Exception as e:
        with open(failed_txt, "w") as f:
            f.write("")
        print(f"Error occurred in result. {e}")
        continue
