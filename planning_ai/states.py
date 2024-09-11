import operator
from pathlib import Path
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document


class OverallState(TypedDict):
    contents: List[str]
    filenames: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str
    filename: Path
