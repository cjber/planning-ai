from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel

from planning_ai.chains.hallucination_chain import HallucinationChecker
from planning_ai.common.utils import filename_reducer


class DocumentState(TypedDict):
    document: Document
    filename: Path

    entities: list[dict]
    themes: set[str]
    summary: BaseModel
    hallucination: HallucinationChecker

    iteration: int


class OverallState(TypedDict):
    executive: str
    documents: Annotated[list[DocumentState], filename_reducer]
    policies_support: str
    policies_object: str

    n_docs: int
