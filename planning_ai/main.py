import logging
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

load_dotenv()


def read_docs():
    df = (
        pl.read_parquet(Paths.STAGING / "gcpt3_testing.parquet")
        .drop_nulls(subset="text")
        .drop("index")
    )
    pdf_ids = [
        int(pdf.stem) if pdf.stem.isdigit() else 0
        for pdf in (Paths.STAGING / "pdfs_azure").glob("*.pdf")
    ]
    pdf_loader = PyPDFDirectoryLoader(Paths.STAGING / "pdfs_azure", silent_errors=True)
    out = pdf_loader.load()

    pdfs_combined = {}
    for page in out:
        id = Path(page.metadata["source"]).stem
        if id in pdfs_combined:
            pdfs_combined[id] = pdfs_combined[id] + page.page_content
        else:
            pdfs_combined[id] = page.page_content

    pdfs_combined = (
        pl.from_dict(pdfs_combined)
        .transpose(include_header=True)
        .rename({"column": "attachments_id", "column_0": "pdf_text"})
        .with_columns(pl.col("attachments_id").cast(int))
    )

    df = (
        df.filter(
            (
                pl.col("representations_document")
                == "Greater Cambridge Local Plan Preferred Options"
            )
            & (pl.col("attachments_id").is_in(pdf_ids))
        )
        .unique("id")
        .with_row_index()
    )
    df = df.join(pdfs_combined, on="attachments_id").with_columns(
        (pl.col("text") + "\n\n" + pl.col("pdf_text")).str.slice(0, 50_000)
    )
    loader = PolarsDataFrameLoader(df, page_content_column="text")

    return list(
        {
            doc.page_content: {"document": doc, "filename": doc.metadata["id"]}
            for doc in loader.load()
            if doc.page_content and len(doc.page_content.split(" ")) > 25
        }.values()
    )


def main():
    docs = read_docs()[:200]
    n_docs = len(docs)

    logging.warning(f"{n_docs} documents being processed!")
    app = create_graph()

    step = None
    for step in app.stream({"documents": docs, "n_docs": n_docs}):
        print(step.keys())

    if step is None:
        raise ValueError("No steps were processed!")

    build_final_report(doc_title, step)
    build_summaries_document(step)
    return step


if __name__ == "__main__":
    doc_title = "Cambridge Response Summary"

    tic = time.time()
    out = main()
    toc = time.time()

    print(f"Time taken: {(toc - tic) / 60:.2f} minutes.")
