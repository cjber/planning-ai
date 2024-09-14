import operator
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document

from planning_ai.chains.hallucination_chain import HallucinationChecker
from planning_ai.chains.map_chain import BriefSummary


class OverallState(TypedDict):
    documents: list[str]
    summaries: Annotated[list, operator.add]
    summaries_fixed: Annotated[list, operator.add]
    hallucinations: Annotated[list, operator.add]

    filenames: List[Path]
    summary_documents: Annotated[list[Document], operator.add]
    final_summary: str

    iterations: list[int]


class DocumentState(TypedDict):
    document: str
    summary: BriefSummary
    hallucination: HallucinationChecker
    filename: Path

    iteration: int
