from pathlib import Path

import polars as pl

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
