import operator
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel

from planning_ai.chains.hallucination_chain import HallucinationChecker
from planning_ai.common.utils import filename_reducer


class DocumentState(TypedDict):
    """Represents the state of an individual document during processing.

    This class is a TypedDict that encapsulates the state of a single document
    during the processing workflow. It includes the document text, summary,
    hallucination details, filename, and iteration count.

    Attributes:
        document (str): The text of the document.
        summary (BriefSummary): The summary of the document.
        hallucination (HallucinationChecker): The hallucination details for the document's summary.
        filename (Path): The file path of the document.
        iteration (int): The current iteration count for processing the document.
    """

    document: Document
    filename: Path

    entities: list[dict]
    themes: set[str]
    summary: BaseModel
    theme_docs: list[Document]
    hallucination: HallucinationChecker

    iteration: int


class OverallState(TypedDict):
    """Represents the overall state of document processing.

    This class is a TypedDict that encapsulates the overall state of the document
    processing workflow. It includes information about the documents, summaries,
    hallucinations, filenames, and iterations.

    Attributes:
        documents (list[str]): A list of document texts.
        final_summary (str): The final aggregated summary of all documents.
        summaries (Annotated[list, operator.add]): A list of initial summaries for each document.
        summaries_fixed (Annotated[list, operator.add]): A list of summaries after fixing hallucinations.
        hallucinations (Annotated[list, operator.add]): A list of detected hallucinations in summaries.
        filenames (List[Path]): A list of file paths corresponding to the documents.
        iterations (list[int]): A list of iteration counts for processing each document.
    """

    documents: Annotated[list[DocumentState], filename_reducer]
    n_docs: int

    final_summary: Optional[str]

    # documents_related: Annotated[list, operator.add]
    # related_theme_docs: Annotated[list, operator.add]
    #
    # summaries: Annotated[list, operator.add]
    # summaries_fixed: Annotated[list, operator.add]
    # hallucinations: Annotated[list, operator.add]
    #
    # filenames: Annotated[list, operator.add]
    # iterations: list[int]
