from pathlib import Path

import polars as pl

pl.Config(
    fmt_str_lengths=9,
    set_tbl_rows=5,
    set_tbl_hide_dtype_separator=True,
    set_tbl_dataframe_shape_below=True,
    set_tbl_formatting="UTF8_FULL_CONDENSED",
)


def filename_reducer(docs_a, docs_b):
    if docs_a == []:
        return docs_b
    b_dict = {d["filename"]: d for d in docs_b}

    for i, dict_a in enumerate(docs_a):
        filename = dict_a.get("filename")
        if filename in b_dict:
            docs_a[i] = b_dict[filename]
    return docs_a


class Paths:
    DATA = Path("data")

    RAW = DATA / "raw"
    STAGING = DATA / "staging"
    OUT = DATA / "out"

    SUMMARY = OUT / "summary"
    SUMMARIES = OUT / "summaries"

    PROMPTS = Path("planning_ai/chains/prompts")

    @classmethod
    def ensure_directories_exist(cls):
        for path in [
            cls.DATA,
            cls.RAW,
            cls.STAGING,
            cls.OUT,
            cls.SUMMARY,
            cls.SUMMARIES,
        ]:
            path.mkdir(parents=True, exist_ok=True)


Paths.ensure_directories_exist()
