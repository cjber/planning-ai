import polars as pl

from planning_ai.common.utils import Paths


def main():
    df = pl.read_excel(
        Paths.RAW / "gclp-first-proposals-questionnaire-responses-redacted.xlsx"
    )

    free_cols = [df.columns[0]] + df.columns[6:13] + [df.columns[33]]
    df = df[free_cols]

    for row in df.rows(named=True):
        user = row.pop("UserNo")
        content = "\n\n".join([f"**{k}**\n\n{v}" for k, v in row.items() if v != "-"])
        with open(Paths.STAGING / "gclp" / f"{user}.txt", "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
