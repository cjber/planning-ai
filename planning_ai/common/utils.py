from pathlib import Path
from typing import List

import polars as pl
from langchain_core.documents import Document

from planning_ai.llms.llm import LLM

pl.Config(
    fmt_str_lengths=9,
    set_tbl_rows=5,
    set_tbl_hide_dtype_separator=True,
    set_tbl_dataframe_shape_below=True,
    set_tbl_formatting="UTF8_FULL_CONDENSED",
)


class Paths:
    DATA = Path("data")
    RAW = DATA / "raw"
    STAGING = DATA / "staging"
    OUT = DATA / "out"


class Consts:
    TOKEN_MAX = 10_000


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(LLM.get_num_tokens(doc.page_content) for doc in documents)
