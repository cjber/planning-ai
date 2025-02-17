import logging
import re
from collections import Counter

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Patch
from polars.dependencies import subprocess

from planning_ai.common.utils import Paths

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{libertine}"

WARDS = [
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


def _process_postcodes(final):
    documents = final["documents"]
    postcodes = [doc["document"].metadata["respondentpostcode"] for doc in documents]
    postcodes = (
        pl.DataFrame({"postcode": postcodes})["postcode"]
        .value_counts()
        .with_columns(pl.col("postcode").str.replace_all(" ", ""))
    )
    onspd = pl.read_parquet(
        Paths.RAW / "onspd_cambridge.parquet",
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
            f"**{item}**: {stance['percentage']:.1%} _({stance['count']})_"
            for item, stance in stances_top
        ]
    )


def _process_themes(final):
    documents = final["documents"]
    themes = Counter(
        [theme["theme"].value for doc in documents for theme in doc["themes"]]
    )
    themes = pl.DataFrame(themes).transpose(include_header=True)
    themes_breakdown = themes.with_columns(
        ((pl.col("column_0") / pl.sum("column_0")) * 100).round(2).alias("percentage")
    ).sort("percentage", descending=True)
    themes_breakdown = themes_breakdown.rename(
        {"column": "Theme", "column_0": "Count", "percentage": "Percentage"}
    )
    pd.set_option("display.precision", 1)
    return themes_breakdown.to_pandas().to_markdown(index=False)


def fig_oa(postcodes):
    oa_lookup = pl.read_csv(
        Paths.RAW
        / "Output_Area_to_Local_Authority_District_(April_2023)_Lookup_in_England_and_Wales.csv"
    )
    camb_oa = oa_lookup.filter(
        pl.col("LAD23NM").is_in(["Cambridge", "South Cambridgeshire"])
    )
    oa_pop = pl.read_csv(Paths.RAW / "oa_populations.csv")
    oa_pop = (
        oa_pop.join(camb_oa, left_on="Output Areas Code", right_on="OA21CD")
        .group_by(pl.col("Output Areas Code"))
        .sum()
        .rename({"Output Areas Code": "OA2021", "Observation": "population"})
        .select(["OA2021", "population"])
    )

    oac = pl.read_csv(Paths.RAW / "oac21ew.csv")
    oac_names = pl.read_csv(Paths.RAW / "classification_codes_and_names.csv")
    oac = (
        oac.with_columns(pl.col("supergroup").cast(str))
        .join(oac_names, left_on="supergroup", right_on="Classification Code")
        .select(["oa21cd", "Classification Name", "supergroup"])
        .rename(
            {
                "Classification Name": "supergroup_name",
            }
        )
    )
    oac = oac.join(oa_pop, left_on="oa21cd", right_on="OA2021")
    oac = (
        postcodes.join(oac, left_on="OA21", right_on="oa21cd", how="right")
        .group_by(["supergroup", "supergroup_name"])
        .sum()
        .select(["supergroup", "supergroup_name", "population", "count"])
        .sort("supergroup")
        .with_columns(
            ((pl.col("count") / pl.col("count").sum()) * 100).alias("perc_count"),
            ((pl.col("population") / pl.col("population").sum()) * 100).alias(
                "perc_pop"
            ),
        )
        .with_columns((pl.col("perc_count") - pl.col("perc_pop")).alias("perc_diff"))
    )
    oa_pd = oac.to_pandas()

    _, ax1 = plt.subplots(figsize=(8, 8))

    # Define a list of colors for each supergroup
    colors = [
        "#7f7f7f",  # retired
        "#2ca02c",  # suburbanites
        "#d62728",  # multicultural
        "#e377c2",  # low skilled
        "#ff7f0e",  # ethnically diverse
        "#bcbd22",  # baseline uk
        "#1f77b4",  # semi unskilled
        "#9467bd",  # legacy
    ]

    # Plot bars for percentage of representations
    bars1 = ax1.bar(
        oa_pd["supergroup"],
        oa_pd["perc_diff"],
        label="Percentage of Representations (\%)",
        color=colors[: len(oa_pd)],
        edgecolor="black",
    )

    # Add centerline at y=0
    ax1.axhline(0, color="black", linewidth=1.5)

    # Annotate bars with percentage values
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(
                f"{height:.0f}\%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        else:
            ax1.annotate(
                f"{height:.0f}\%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -6),  # 10 points vertical offset
                textcoords="offset points",
                ha="center",
                va="top",
            )

    ax1.set_xlabel("Output Area Classification (OAC) Supergroup")
    ax1.set_ylabel("Difference from national average (\%)")

    supergroup_names = [
        f"{i}: {name}"
        for i, name in enumerate(oa_pd["supergroup_name"].unique(), start=1)
    ]
    legend_patches = [
        Patch(color=colors[i], label=supergroup_names[i])
        for i in range(len(supergroup_names))
    ]
    ax1.legend(handles=legend_patches, title="Supergroup", frameon=False)

    plt.tight_layout()

    plt.savefig(Paths.SUMMARY / "figs" / "oas.pdf")


def fig_wards(postcodes):
    ward_boundaries = gpd.read_file(
        Paths.RAW / "Wards_December_2021_GB_BFE_2022_7523259277605796091.zip"
    )

    camb_ward_boundaries = ward_boundaries[ward_boundaries["WD21CD"].isin(WARDS)]
    ward_boundaries_prop = ward_boundaries.merge(
        postcodes.to_pandas(), left_on="WD21CD", right_on="OSWARD"
    )

    _, ax = plt.subplots(figsize=(8, 8))
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
    buffer = 20_000
    ax.set_xlim([bounds[0] - buffer, bounds[2] + buffer])
    ax.set_ylim([bounds[1] - buffer, bounds[3] + buffer])

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(Paths.SUMMARY / "figs" / "wards.pdf")


def fig_imd(postcodes):
    imd = pl.read_csv(
        Paths.RAW / "uk_imd2019.csv", columns=["LSOA", "SOA_decile"]
    ).with_columns(((pl.col("SOA_decile") - 1) // 2) + 1)

    oa_lookup = pl.read_csv(Paths.RAW / "lsoa_lad_lookup.csv")[
        ["LSOA11CD", "LAD11NM"]
    ].unique()
    lsoa_camb = oa_lookup.filter(
        pl.col("LAD11NM").is_in(["Cambridge", "South Cambridgeshire"])
    )

    imd = imd.join(lsoa_camb, left_on="LSOA", right_on="LSOA11CD")
    pops = pl.read_excel(
        Paths.RAW / "sapelsoabroadage20112022.xlsx",
        sheet_name="Mid-2022 LSOA 2021",
        read_options={"header_row": 3},
        columns=["LSOA 2021 Code", "Total"],
    )
    imd = (
        postcodes.join(imd, left_on="LSOA11", right_on="LSOA", how="right")
        .join(pops, left_on="LSOA", right_on="LSOA 2021 Code")
        .group_by("SOA_decile")
        .agg(pl.col("count").sum(), pl.col("LSOA").count(), pl.col("Total").sum())
        .sort("SOA_decile")
        .with_columns(
            ((pl.col("count") / pl.col("count").sum()) * 100).alias("perc_count"),
            ((pl.col("Total") / pl.col("Total").sum()) * 100).alias("perc_pop"),
        )
        .with_columns((pl.col("perc_count") - pl.col("perc_pop")).alias("perc_diff"))
    )

    postcodes_pd = imd.to_pandas()
    colors = [
        "#d62728",
        "#9f4e64",
        "#6f76a0",
        "#478dbf",
        "#1f77b4",
    ]

    _, ax1 = plt.subplots(figsize=(8, 8))

    x = np.arange(len(postcodes_pd))
    ax1.bar(
        x,  # Shift to the left
        postcodes_pd["perc_diff"],
        edgecolor="black",
        color=colors,
    )

    # Set labels and ticks
    ax1.set_xlabel("Deprivation Quintile")
    ax1.set_ylabel("Difference from national average (\%)")
    ax1.set_xticks(x)
    ax1.axhline(0, color="black", linewidth=1.5)

    # ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False)
    plt.tight_layout()
    ax1.set_xticklabels(["1 - Most Deprived", "2", "3", "4", "5 - Least Deprived"])

    plt.savefig(Paths.SUMMARY / "figs" / "imd_decile.pdf")


def build_final_report(out, rep):
    introduction_paragraph = """
This report was produced using a generative pre-trained transformer (GPT) large-language model (LLM) to produce an abstractive summary of all responses to the related planning application. This model automatically reviews every response in detail, and extracts key information to inform decision making. This document first consolidates this information into a single-page executive summary, highlighting areas of particular interest to consider, and the broad consensus of responses. Figures generated from responses then give both a geographic and statistical overview, highlighting any demographic imbalances in responses. The document then extracts detailed information from responses, grouped by theme and policy. In this section we incorporate citations which relate with the 'Summary Responses' document, to increase transparency.
"""
    figures_paragraph = r"""
This section describes the characteristics of where submissions were received from. This can help to identify how representative submissions were and whether there were any communities whose views were not being considered. Figure \ref{fig-wards} shows the number (frequency) of submitted representations by Ward based on the address attached to the submission. To interpret the figure, areas which are coloured white had no submissions from residents, and then areas are coloured in based on the total number of submissions with yellows and greens representing the largest numbers. This figure helps to identify which Wards are more active in terms of participation and representation in this report.

Figure \ref{fig-oas} displays the percentage of representations submitted by the Output Area Classification (2021). The Output Area Classification is the Office for National Statistics preferred classification of neighbourhoods.  This measure groups neighbourhoods (here defined as Output Areas, typically containing 100 people) into categories that capture similar types of people based on population, demographic and socioeconomic characteristics. It therefore provides an insightful view of the types of communities who submitted representations. To interpret the figure, where bars extend higher/upwards, this represents a larger population share within a specific area type. The blue bars represent the characteristics of who submitted representations, and the orange bars represent the underlying population – allowing one to compare whether the profile of submissions matched the characteristics of the local population. This figure uses OAC 'Supergroups', which are the highest level of the hierarchy, and provide information relative to the average values for the UK population at large.

Figure \ref{fig-imd} shows the percentage of responses by level of neighbourhood socioeconomic deprivation. The information is presented using the 2019 Index of Multiple Deprivation, divided into quintiles (i.e., dividing the English population into equal fifths). This measure is the UK Government’s preferred measure of socioeconomic deprivation and is based on information about income, employment, education, health, crime, housing and the local environment for small areas (Lower Super Output Areas, typically containing 1600 people). To interpret the graph, bars represent the share of population from each quintile. Quintile 1 represents the most deprived 20% of areas, and quintile 5 the least deprived 20% of areas. The orange bars represent the distribution of people who submitted representations (i.e., larger bars mean that more people from these areas submitted representations). The blue bars show the distribution of the local population, allowing one to evaluate whether the evidence submitted was from the same communities in the area.
"""
    themes_paragraph = """
The following section provides a detailed breakdown of notable details from responses, grouped by themes and policies. Both themes and associated policies are automatically determined through an analysis of the summary content by an LLM agent. Each theme is grouped by whether a responses is supporting, opposed, or a general comment. This section aims to give a comprehensive view of the key issues raised by the respondents with respect to the themes and policies outlined. We have incorporated citations into eac hpoint (see numbers in square brackets) which relate to the specific document they were made in, to promote the transparency of where information was sourced from. @tbl-themes gives a breakdown of the number of submissions that relate with each theme, submissions may relate to more than one theme.
    """
    final = out["generate_final_report"]
    unused_documents = out["generate_final_report"]["unused_documents"]
    support_policies, object_policies, other_policies = _process_policies(final)
    postcodes = _process_postcodes(final)
    stances = _process_stances(final)
    themes = _process_themes(final)

    fig_wards(postcodes)
    fig_oa(postcodes)
    fig_imd(postcodes)

    quarto_doc = (
        "---\n"
        f"title: 'Summary of Submitted Representations: {rep}'\n"
        "geometry: a4paper\n"
        "margin: 2cm\n"
        "fontfamily: libertinus\n"
        "monofont: 'JetBrains Mono'\n"
        "monofontoptions:\n"
        "  - Scale=0.55\n"
        "---\n\n"
        "# Executive Summary\n\n"
        f"{final['executive']}\n\n"
        f"There were a total of {len(final['documents']):,} responses. Of these, submissions indicated "
        "the following support and objection of the plan:\n\n"
        f"{stances}\n\n"
        "# Introduction\n\n"
        f"{introduction_paragraph}\n\n"
        "\n# Profile of Submissions\n\n"
        f"{figures_paragraph}\n\n"
        f"![Total number of representations submitted by Ward\\label{{fig-wards}}](./data/out/summary/figs/wards.pdf)\n\n"
        f"![Total number of representations submitted by Output Area (OA 2021)\\label{{fig-oas}}](./data/out/summary/figs/oas.pdf)\n\n"
        f"![Percentage of representations submitted by quintile of index of multiple deprivation (2019)\\label{{fig-imd}}](./data/out/summary/figs/imd_decile.pdf)\n\n"
        r"\newpage"
        "\n\n# Themes and Policies\n\n"
        f"{themes_paragraph}\n\n"
        f"{themes}{{#tbl-themes}}\n\n"
        "## Supporting Representations\n\n"
        "The following section presents a list of all points raised in representations that support the plan"
        ", grouped by theme and policy.\n\n"
        f"{support_policies or '_No supporting representations._'}\n\n"
        "## Objecting Representations\n\n"
        "The following section presents a list of all points raised in representations that object to "
        "the plan, grouped by theme and policy.\n\n"
        f"{object_policies or '_No objecting representations._'}\n\n"
        "## Comment\n\n"
        "The following section presents a list of all points raised in representations that do not support "
        "or object to the plan, grouped by theme and policy.\n\n"
        f"{other_policies or '_No other representations._'}\n\n"
        "## Unused Documents\n\n"
        "Please note that the following documents were not used to produce this report:\n\n"
        f"{str(unused_documents)}"
    )

    out_path = Paths.SUMMARY / f"Summary_of_Submitted_Responses-{rep}.md"
    out_file = Paths.SUMMARY / f"Summary_of_Submitted_Responses-{rep}.pdf"
    with open(out_path, "w") as f:
        f.write(quarto_doc)
    command = ["pandoc", f"{out_path}", "-o", f"{out_file}"]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during Summary_of_Submitted_Responses.md render: {e}")


def build_summaries_document(out, rep):
    sub = r"Document ID: \[\d+\]\n\n"
    full_text = "".join(
        f"**Document ID**: {document['doc_id']}\n\n"
        # f"**Original Document**\n\n{document['document'].page_content}\n\n"
        f"**Summarised Document**\n\n{re.sub(sub, '', document['summary'].summary)}\n\n"
        # f"**Identified Entities**\n\n{document['entities']}\n\n"
        for document in out["generate_final_report"]["documents"]
    )
    header = (
        "---\n"
        f"title: 'Summary Documents: {rep}'\n"
        "fontfamily: libertinus\n"
        "geometry: a4paper\n"
        "margin: 2cm\n"
        "monofont: 'JetBrains Mono'\n"
        "monofontoptions:\n"
        "  - Scale=0.55\n"
        "---\n\n"
    )
    out_path = Paths.SUMMARY / f"Summary_Documents-{rep}.md"
    out_file = Paths.SUMMARY / f"Summary_Documents-{rep}.pdf"
    with open(out_path, "w") as f:
        f.write(f"{header}{full_text}")

    command = ["pandoc", f"{out_path}", "-o", f"{out_file}"]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during render: {e}")
