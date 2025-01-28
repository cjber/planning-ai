from pathlib import Path
from typing import Annotated, TypedDict

import polars as pl
from langchain_core.documents import Document
from pydantic import BaseModel

from planning_ai.chains.hallucination_chain import HallucinationChecker
from planning_ai.common.utils import filename_reducer


class DocumentState(TypedDict):
    document: Document
    filename: int

    entities: list[dict]
    themes: set[str]

    summary: BaseModel
    hallucination: HallucinationChecker

    is_hallucinated: bool
    refinement_attempts: int
    failed: bool
    processed: bool


class OverallState(TypedDict):
    documents: Annotated[list, filename_reducer]
    executive: str
    policies: pl.DataFrame

    n_docs: int
