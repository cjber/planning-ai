import polars as pl

from planning_ai.common.utils import Paths

dfs = pl.read_excel(Paths.RAW / "web comments.xlsx", sheet_id=0)

for sheet_name, df in dfs.items():
    string_df = df.select(pl.col(pl.String)).drop_nulls()
    for col in string_df.columns:
        series = string_df[col]
        name = series.name
        content = f"**{name}**" + "\n\n* ".join(["\n"] + series.to_list())
        with open(Paths.STAGING / "web" / f"{sheet_name}.txt", "w") as f:
            f.write(content)
