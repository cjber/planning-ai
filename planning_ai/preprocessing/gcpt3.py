import logging
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import requests
from pypdf import PdfReader
from tqdm import tqdm

from planning_ai.common.utils import Paths


def get_schema() -> dict[str, Any]:
    return {
        "id": pl.Int64,
        "method": pl.String,
        "text": pl.String,
        "respondentpostcode": pl.String,
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


def download_attachments():
    df = pl.read_parquet(Paths.STAGING / "gcpt3.parquet")

    existing_files = {f.stem for f in (Paths.RAW / "pdfs").glob("*.pdf")}

    failed_files = set()
    failed_file_path = Paths.RAW / "failed_downloads.txt"
    if failed_file_path.exists():
        with open(failed_file_path, "r") as file:
            failed_files = set(file.read().splitlines())

    for row in tqdm(
        df.drop_nulls(subset="attachments_id")
        .unique(subset="attachments_id")
        .sample(shuffle=True, fraction=1)
        .rows(named=True)
    ):
        if (
            row["attachments_url"].startswith(
                ("https://egov.scambs.gov.uk", "http://egov.scambs.gov.uk")
            )
            or row["attachments_id"] in existing_files
            or row["attachments_id"] in failed_files
        ):
            failed_files.add(row["attachments_id"])
            continue
        file_path = Paths.RAW / "pdfs" / f"{row['attachments_id']}.pdf"
        try:
            response = requests.get(row["attachments_url"], timeout=3)
            response.raise_for_status()

            PdfReader(BytesIO(response.content))  # check if pdf is valid
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {row['attachments_url']} to {file_path}")

        except requests.RequestException as e:
            logging.error(f"RequestException for {row['attachments_url']}: {e}")
            failed_files.add(row["attachments_id"])
            with open(failed_file_path, "a") as file:
                file.write(f"{row['attachments_id']}\n")
            print(f"Skipping {row['attachments_url']} due to error: {e}")

        except Exception as e:
            logging.error(f"Unexpected error for {row['attachments_url']}: {e}")
            row["attachments_url"]
            failed_files.add(row["attachments_id"])
            with open(failed_file_path, "a") as file:
                file.write(f"{row['attachments_id']}\n")
            print(f"Unexpected error for {row['attachments_url']}: {e}")


def main() -> None:
    files = list(Path(Paths.RAW / "gcpt3").glob("*.json"))
    schema = get_schema()
    process_files(files, schema)
    download_attachments()


if __name__ == "__main__":
    main()
