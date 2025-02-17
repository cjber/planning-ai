import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PolarsDataFrameLoader,
    PyPDFDirectoryLoader,
)

from planning_ai.common.utils import Paths
from planning_ai.documents.document import build_final_report, build_summaries_document
from planning_ai.graph import create_graph
from planning_ai.logging import logger

load_dotenv()


def read_docs(representations_document: str):
    logger.warning("Reading documents...")
    df = (
        pl.read_parquet(Paths.STAGING / "gcpt3.parquet")
        .drop_nulls(subset="text")
        .filter(pl.col("representations_document") == representations_document)
    )
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
        # for now I concat page number to keep all pdf pages separate. I might want
        # to instead combine pdfs somehow
        pdf.metadata["filename"] = int(f"{pdf.metadata['id']}999{pdf.metadata['page']}")

    df = df.unique("id").with_columns(filename=pl.col("id"))

    loader = PolarsDataFrameLoader(df, page_content_column="text")
    logger.warning("Loading text files...")
    text = loader.load()
    out = text + pdfs

    # removes duplicates documents based on page_content
    docs = list(
        {
            doc.page_content: doc
            for doc in out
            if doc.page_content and len(doc.page_content.split(" ")) > 25
        }.values()
    )
    return [{"document": doc, "filename": doc.metadata["filename"]} for doc in docs]


def main():
    representations_documents = (
        pl.read_parquet(Paths.STAGING / "gcpt3.parquet")["representations_document"]
        .unique()
        .to_list()
    )
    for rep in representations_documents:
        docs = read_docs(rep)
        n_docs = len(docs)

        logger.info(f"{n_docs} documents being processed!")
        app = create_graph()

        step = None
        for step in app.stream({"documents": docs, "n_docs": n_docs}):
            print(step.keys())

        if step is None:
            raise ValueError("No steps were processed!")

        build_final_report(step, rep)
        build_summaries_document(step, rep)

    return representations_documents


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()

    print(f"Time taken: {(toc - tic) / 60:.2f} minutes.")
