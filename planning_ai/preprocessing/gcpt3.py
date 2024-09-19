from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from planning_ai.common.utils import Paths


def get_schema() -> dict[str, Any]:
    return {
        "id": pl.Int64,
        "method": pl.Utf8,
        "text": pl.Utf8,
        "attachments": pl.List(
            pl.Struct(
                [
                    pl.Field("id", pl.Int64),
                    pl.Field("url", pl.String),
                    pl.Field("published", pl.Boolean),
                ]
            )
        ),
        "representations": pl.List(
            pl.Struct(
                [
                    pl.Field("id", pl.Int64),
                    pl.Field("support/object", pl.Utf8),
                    pl.Field("document", pl.String),
                    pl.Field("documentelementid", pl.Int64),
                    pl.Field("documentelementtitle", pl.String),
                    pl.Field("summary", pl.String),
                ]
            )
        ),
    }


def process_files(files: list[Path], schema: dict[str, Any]) -> None:
    dfs = [pl.read_json(file, schema=schema) for file in tqdm(files)]
    (
        pl.concat(dfs)
        .explode("attachments")
        .explode("representations")
        .with_columns(
            pl.col("attachments").name.map_fields(lambda x: f"attachments_{x}")
        )
        .unnest("attachments")
        .with_columns(
            pl.col("representations").name.map_fields(lambda x: f"representations_{x}")
        )
        .unnest("representations")
        .write_parquet(Paths.STAGING / "gcpt3.parquet")
    )


def main() -> None:
    files = list(Path(Paths.RAW / "gcpt3").glob("*.json"))
    schema = get_schema()
    process_files(files, schema)


if __name__ == "__main__":
    main()
