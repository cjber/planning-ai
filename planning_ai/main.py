import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PolarsDataFrameLoader,
    PyPDFDirectoryLoader,
)

from planning_ai.common.utils import Paths
from planning_ai.document import build_final_report, build_summaries_document
from planning_ai.graph import create_graph
from planning_ai.logging import logger

load_dotenv()


def read_docs():
    logger.warning("Reading documents...")
    df = (
        pl.read_parquet(Paths.STAGING / "gcpt3_testing.parquet")
        .drop_nulls(subset="text")
        .drop("index")
    )
    pdf_ids = [
        int(pdf.stem) if pdf.stem.isdigit() else 0
        for pdf in (Paths.STAGING / "pdfs_azure").glob("*.pdf")
    ]
    pdf_loader = PyPDFDirectoryLoader(
        (Paths.STAGING / "pdfs_azure"), silent_errors=True
    )

    logger.warning("Loading PDFs...")
    pdfs = pdf_loader.load()

    for pdf in pdfs:
        pdf.metadata["id"] = Path(pdf.metadata["source"]).stem
        meta = (
            df.filter(pl.col("attachments_id") == int(pdf.metadata["id"]))
            .select(["respondentpostcode", "representations_support/object"])
            .to_dict(as_series=False)
        )
        pdf.metadata = pdf.metadata | {
            "respondentpostcode": (
                meta["respondentpostcode"][0] if meta["respondentpostcode"] else ""
            ),
            "representations_support/object": (
                meta["representations_support/object"][0]
                if meta["representations_support/object"]
                else ""
            ),
        }

    df = df.filter(
        (
            pl.col("representations_document")
            == "Greater Cambridge Local Plan Preferred Options"
        )
        & (pl.col("attachments_id").is_in(pdf_ids))
    ).unique("id")

    loader = PolarsDataFrameLoader(df, page_content_column="text")
    logger.warning("Loading text files...")
    text = loader.load()
    out = text + pdfs

    return list(
        {
            doc.page_content: {"document": doc, "filename": doc.metadata["id"]}
            for doc in out
            if doc.page_content and len(doc.page_content.split(" ")) > 25
        }.values()
    )


def main():
    docs = read_docs()[:500]
    n_docs = len(docs)

    logger.info(f"{n_docs} documents being processed!")
    app = create_graph()

    step = None
    for step in app.stream({"documents": docs, "n_docs": n_docs}):
        print(step.keys())

    step["generate_final_report"]
    if step is None:
        raise ValueError("No steps were processed!")

    return step


if __name__ == "__main__":
    doc_title = "Cambridge Response Summary"

    tic = time.time()
    out = main()
    build_final_report(doc_title, out)
    build_summaries_document(out)
    toc = time.time()

    print(f"Time taken: {(toc - tic) / 60:.2f} minutes.")
