import itertools
import logging
import re
from collections import Counter

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars.dependencies import subprocess

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
        Paths.RAW / "onspd" / "ONSPD_FEB_2024.csv",
        columns=["PCD", "OSWARD", "LSOA11", "OA21"],
    ).with_columns(pl.col("PCD").str.replace_all(" ", "").alias("postcode"))
    postcodes = postcodes.join(onspd, on="postcode")
    return postcodes


def _process_policies(final):
    def process_policy_group(policy_group, theme, stance):
        details = "".join(
            f'\n### {row["policies"]}\n\n'
            + "".join(
                f"- {detail} {doc_id}\n"
                for detail, doc_id in zip(row["detail"], row["doc_id"])
            )
            for row in policy_group.rows(named=True)
        )
        return f"## {theme} - {stance}\n\n{details}\n"

    policies_df = final["policies"]

    support_policies = ""
    object_policies = ""
    other_policies = ""

    for (theme, stance), policy in policies_df.group_by(
        ["themes", "stance"], maintain_order=True
    ):
        if stance == "Support":
            support_policies += process_policy_group(policy, theme, stance)
        elif stance == "Object":
            object_policies += process_policy_group(policy, theme, stance)
        else:
            other_policies += process_policy_group(policy, theme, stance)

    return support_policies, object_policies, other_policies


def _process_stances(final):
    documents = final["documents"]
    stances = [
        doc["document"].metadata["representations_support/object"] for doc in documents
    ]
    value_counts = Counter(stances)
    total_values = sum(value_counts.values())
    percentages = {
        key: {"count": count, "percentage": (count / total_values)}
        for key, count in value_counts.items()
    }
    stances_top = sorted(
        percentages.items(), key=lambda x: x[1]["percentage"], reverse=True
    )
    return " | ".join(
        [
            f"**{item}**: {stance['percentage']:.2%} _({stance['count']})_"
            for item, stance in stances_top
        ]
    )


def _process_themes(final):
    documents = final["documents"]
    themes = [list(doc["themes"]) for doc in documents]
    themes = Counter(list(itertools.chain.from_iterable(themes)))
    themes = pl.DataFrame(themes).transpose(include_header=True)
    themes_breakdown = themes.with_columns(
        ((pl.col("column_0") / pl.sum("column_0")) * 100).round(2).alias("percentage")
    ).sort("percentage", descending=True)
    themes_breakdown = themes_breakdown.rename(
        {"column": "Theme", "column_0": "Count", "percentage": "Percentage"}
    )
    return themes_breakdown.to_pandas().to_markdown(index=False)


def fig_oa(postcodes):
    oac = pl.read_csv(Paths.RAW / "oac21ew.csv")
    postcodes = (
        postcodes.join(oac, left_on="OA21", right_on="oa21cd")
        .group_by("supergroup")
        .len()
        .sort("supergroup")
    )
    postcodes_pd = postcodes.to_pandas()

    _, ax1 = plt.subplots()

    ax1.bar(postcodes_pd["supergroup"], postcodes_pd["len"])
    ax1.set_xlabel("Output Area Classification (OAC) Supergroup")
    ax1.set_ylabel("Number of Representations")

    plt.tight_layout()

    plt.savefig(Paths.SUMMARY / "figs" / "oas.png")


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
    ward_boundaries_prop.plot(
        ax=ax,
        column="count",
        legend=True,
        vmax=20,
        legend_kwds={"label": "Number of Representations"},
    )
    ward_boundaries.plot(ax=ax, color="none", edgecolor="gray")
    camb_ward_boundaries.plot(ax=ax, color="none", edgecolor="black")

    bounds = np.array([541419.8982, 253158.2036, 549420.4025, 262079.7998])
    buffer = 10_000
    ax.set_xlim([bounds[0] - buffer, bounds[2] + buffer])
    ax.set_ylim([bounds[1] - buffer, bounds[3] + buffer])

    plt.axis("off")
    plt.tight_layout()

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
        .with_columns(
            ((pl.col("count") / pl.col("count").sum()) * 100).alias("perc_count"),
            ((pl.col("Total") / pl.col("Total").sum()) * 100).alias("perc_pop"),
        )
    )

    postcodes_pd = postcodes.to_pandas()

    _, ax1 = plt.subplots()

    # Define bar width
    bar_width = 0.35

    # Set positions for the bars
    x = np.arange(len(postcodes_pd))

    # Plot the first set of bars
    ax1.bar(
        x - bar_width / 2,  # Shift to the left
        postcodes_pd["perc_count"],
        width=bar_width,
        label="Percentage of Count (%)",
    )

    # Plot the second set of bars
    ax1.bar(
        x + bar_width / 2,  # Shift to the right
        postcodes_pd["perc_pop"],
        width=bar_width,
        label="Percentage of Population (%)",
    )

    # Set labels and ticks
    ax1.set_xlabel("IMD Quintile")
    ax1.set_ylabel("Proportion of Population (%)")
    ax1.set_xticks(x)  # Set x-ticks to correspond to the positions
    ax1.set_xticklabels(postcodes_pd["LA_decile"])

    # Add a legend
    ax1.legend()

    # Adjust layout
    plt.tight_layout()

    plt.savefig(Paths.SUMMARY / "figs" / "imd_decile.png")


def build_final_report(out):
    introduction_paragraph = """
This report was produced using a generative pre-trained transformer (GPT) large-language model (LLM) to produce an abstractive summary of all responses to the related planning application. This model automatically reviews every response in detail, and extracts key information to inform decision making. This document first consolidates this information into a single-page executive summary, highlighting areas of particular interest to consider, and the broad consensus of responses. Figures generated from responses then give both a geographic and statistical overview, highlighting any demographic imbalances in responses. The document then extracts detailed information from responses, grouped by theme and policy. In this section we incorporate citations which relate with the 'Summary Responses' document, to increase transparency.
"""
    figures_paragraph = """
@fig-wards shows the percentage of responses by total population within each Ward that had at least one response. This figure helps to identify which Wards are more active in terms of participation and representation. @fig-imd shows the percentage of responses by total population within each IMD quintile. This figure provides insight into the socio-economic distribution of the respondents, highlighting any potential demographic imbalances. @fig-oas displays the total number of representations submitted by Output Area (OA 2021). This figure offers a detailed geographic overview of the responses, allowing for a more granular analysis of participation across different areas.
"""
    themes_paragraph = """
The following section provides a detailed breakdown of notable details from responses, grouped by themes and policies. Each theme is grouped by whether a responses is supporting, opposed, or a general comment. This section aims to give a comprehensive view of the key issues raised by the respondents with respect to the themes and policies outlined.
    """
    final = out["generate_final_report"]
    support_policies, object_policies, other_policies = _process_policies(final)
    postcodes = _process_postcodes(final)
    stances = _process_stances(final)
    themes = _process_themes(final)

    fig_wards(postcodes)
    fig_oa(postcodes)
    fig_imd(postcodes)

    quarto_doc = (
        "---\n"
        f"title: 'Summary of Submitted Responses'\n"
        "format:\n"
        "  pdf:\n"
        "    papersize: A4\n"
        "execute:\n"
        "  freeze: auto\n"
        "  echo: false\n"
        "fontfamily: libertinus\n"
        "monofont: 'JetBrains Mono'\n"
        "monofontoptions:\n"
        "  - Scale=0.55\n"
        "---\n\n"
        f"{final['executive']}\n\n"
        f"{stances}\n\n"
        "# Introduction\n\n"
        f"{introduction_paragraph}\n\n"
        "\n# Figures\n\n"
        f"{figures_paragraph}\n\n"
        f"![Total number of representations submitted by Ward.](./figs/wards.png){{#fig-wards}}\n\n"
        f"![Total number of representations submitted by Output Area (OA 2021).](./figs/oas.png){{#fig-oas}}\n\n"
        f"![Percentage of representations submitted by quintile of index of multiple deprivation (2019)](./figs/imd_decile.png){{#fig-imd}}\n\n"
        "# Themes and Policies\n\n"
        f"{themes_paragraph}\n\n"
        f"{themes}{{#tbl-themes}}\n\n"
        "## Support\n\n"
        f"{support_policies}\n\n"
        "## Object\n\n"
        f"{object_policies}\n\n"
        "## Other\n\n"
        f"{other_policies}\n\n"
    )

    with open(Paths.SUMMARY / "Summary_of_Submitted_Responses.qmd", "w") as f:
        f.write(quarto_doc)
    command = [
        "quarto",
        "render",
        f"{Paths.SUMMARY / 'Summary_of_Submitted_Responses.qmd'}",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during Summary_of_Submitted_Responses.qmd render: {e}")


def build_summaries_document(out):
    sub = r"Document ID: \[\d+\]\n\n"
    full_text = "".join(
        f"**Document ID**: {document['doc_id']}\n\n"
        # f"**Original Document**\n\n{document['document'].page_content}\n\n"
        f"**Summarised Document**\n\n{re.sub(sub, '', document['summary'].summary)}\n\n"
        # f"**Identified Entities**\n\n{document['entities']}\n\n"
        for document in out["generate_final_report"]["documents"]
    )
    quarto_header = (
        "---\n"
        "title: 'Summary Documents'\n"
        "format:\n"
        "  pdf:\n"
        "    papersize: A4\n"
        "execute:\n"
        "  freeze: auto\n"
        "  echo: false\n"
        "fontfamily: libertinus\n"
        "monofont: 'JetBrains Mono'\n"
        "monofontoptions:\n"
        "  - Scale=0.55\n"
        "---\n\n"
    )
    with open(Paths.SUMMARY / "Summary_Documents.qmd", "w") as f:
        f.write(f"{quarto_header}{full_text}")

    command = ["quarto", "render", f"{Paths.SUMMARY / 'Summary_Documents.qmd'}"]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during Summary_Documents.qmd render: {e}")
