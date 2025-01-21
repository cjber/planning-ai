import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl

from planning_ai.common.utils import Paths


def _process_postcodes(final):
    documents = final["documents"]
    postcodes = [doc["document"].metadata["respondentpostcode"] for doc in documents]
    postcodes = (
        pl.DataFrame({"postcode": postcodes})["postcode"]
        .value_counts()
        .with_columns(pl.col("postcode").str.replace_all(" ", ""))
    )
    onspd = pl.read_csv(
        Paths.RAW / "onspd" / "ONSPD_FEB_2024.csv", columns=["PCD", "OSWARD", "LSOA11"]
    ).with_columns(pl.col("PCD").str.replace_all(" ", "").alias("postcode"))
    postcodes = postcodes.join(onspd, on="postcode")
    return postcodes


def _process_policies(final):
    policies_df = final["policies"]

    all_policies = ""
    for (theme, stance), policy in policies_df.group_by(
        ["themes", "stance"], maintain_order=True
    ):
        details = "".join(
            f'\n### {row["policies"]}\n\n'
            + "".join(
                f"- {detail} {doc_id}\n"
                for detail, doc_id in zip(row["detail"], row["doc_id"])
            )
            for row in policy.rows(named=True)
        )
        all_policies += f"## {theme} - {stance}\n\n{details}\n"
    return all_policies


def fig_wards(postcodes):
    wards = (
        pl.read_csv(Paths.RAW / "TS001-2021-3-filtered-2025-01-09T11_07_15Z.csv")
        .rename(
            {
                "Electoral wards and divisions Code": "OSWARD",
                "Electoral wards and divisions": "WARDNAME",
            }
        )
        .group_by(["OSWARD", "WARDNAME"])
        .sum()
    )
    postcodes = postcodes.join(wards, on="OSWARD").with_columns(
        ((pl.col("count") / pl.col("Observation")) * 100).alias("prop")
    )
    ward_boundaries = gpd.read_file(
        Paths.RAW / "Wards_December_2021_GB_BFE_2022_7523259277605796091.zip"
    )

    camb_ward_codes = [
        "E05013050",
        "E05013051",
        "E05013052",
        "E05013053",
        "E05013054",
        "E05013055",
        "E05013056",
        "E05013057",
        "E05013058",
        "E05013059",
        "E05013060",
        "E05013061",
        "E05013062",
        "E05013063",
    ]
    camb_ward_boundaries = ward_boundaries[
        ward_boundaries["WD21CD"].isin(camb_ward_codes)
    ]
    ward_boundaries_prop = ward_boundaries.merge(
        postcodes.to_pandas(), left_on="WD21CD", right_on="OSWARD"
    )

    _, ax = plt.subplots()
    ward_boundaries.plot(ax=ax, color="white", edgecolor="gray")
    camb_ward_boundaries.plot(ax=ax, color="white", edgecolor="black")
    ward_boundaries_prop.plot(ax=ax, column="prop", legend=True)

    bounds = camb_ward_boundaries.total_bounds
    buffer = 10000
    ax.set_xlim([bounds[0] - buffer, bounds[2] + buffer])
    ax.set_ylim([bounds[1] - buffer, bounds[3] + buffer])

    plt.axis("off")
    plt.savefig(Paths.SUMMARY / "figs" / "wards.png")


def fig_imd(postcodes):
    imd = pl.read_csv(
        Paths.RAW / "uk_imd2019.csv", columns=["LSOA", "LA_decile"]
    ).with_columns(((pl.col("LA_decile") - 1) // 2) + 1)
    pops = pl.read_excel(
        Paths.RAW / "sapelsoabroadage20112022.xlsx",
        sheet_name="Mid-2022 LSOA 2021",
        read_options={"header_row": 3},
        columns=["LSOA 2021 Code", "Total"],
    )

    postcodes = (
        postcodes.join(imd, left_on="LSOA11", right_on="LSOA")
        .join(pops, left_on="LSOA11", right_on="LSOA 2021 Code")
        .group_by("LA_decile")
        .agg(pl.col("count").sum(), pl.col("LSOA11").count(), pl.col("Total").sum())
        .sort("LA_decile")
        .with_columns(((pl.col("count") / pl.col("Total")) * 100).alias("prop"))
    )

    postcodes_pd = postcodes.to_pandas()

    _, ax1 = plt.subplots()

    ax1.bar(
        postcodes_pd["LA_decile"],
        postcodes_pd["prop"],
        label="Percentage of Population (%)",
    )
    ax1.set_xlabel("IMD Quintile")
    ax1.set_ylabel("Proporition of Population (%)")
    ax1.tick_params(axis="y")

    plt.title("Comparison of Responses by IMD Quintile")

    plt.tight_layout()
    plt.savefig(Paths.SUMMARY / "figs" / "imd_decile.png")


def build_final_report(doc_title, out):
    final = out["generate_final_report"]
    policies = _process_policies(final)
    postcodes = _process_postcodes(final)

    fig_wards(postcodes)
    fig_imd(postcodes)

    quarto_doc = (
        "---\n"
        f"title: '{doc_title}'\n"
        "format:\n"
        "  PrettyPDF-pdf:\n"
        "    papersize: A4\n"
        "execute:\n"
        "  freeze: auto\n"
        "  echo: false\n"
        "monofont: 'JetBrains Mono'\n"
        "monofontoptions:\n"
        "  - Scale=0.55\n"
        "---\n\n"
        f"{final['executive']}\n\n"
        "\n# Figures\n\n"
        "@fig-wards shows the percentage of responses by total population"
        " within each Ward that had at least one response.\n\n"
        f"![Ward Proportions](./figs/wards.png){{#fig-wards}}\n\n"
        "@fig-imd shows the percentage of responses by total population"
        " within each IMD quintile.\n\n"
        f"![IMD Quintile Props](./figs/imd_decile.png){{#fig-imd}}\n\n"
        "# Themes and Policies\n\n"
        f"{policies}"
    )

    with open(Paths.SUMMARY / f"{doc_title.replace(' ', '_')}.qmd", "w") as f:
        f.write(quarto_doc)


def build_summaries_document(out):
    full_text = "".join(
        f"**Document ID**: {document['doc_id']}\n\n"
        # f"**Original Document**\n\n{document['document'].page_content}\n\n"
        f"**Summarised Document**\n\n{document['summary'].summary}\n\n"
        # f"**Identified Entities**\n\n{document['entities']}\n\n"
        for document in out["generate_final_report"]["documents"]
    )
    quarto_header = (
        "---\n"
        "title: 'Summary Documents'\n"
        "format:\n"
        "  PrettyPDF-pdf:\n"
        "    papersize: A4\n"
        "execute:\n"
        "  freeze: auto\n"
        "  echo: false\n"
        "monofont: 'JetBrains Mono'\n"
        "monofontoptions:\n"
        "  - Scale=0.55\n"
        "---\n\n"
    )
    with open(Paths.SUMMARY / "Summary_Documents.qmd", "w") as f:
        f.write(f"{quarto_header}{full_text}")
