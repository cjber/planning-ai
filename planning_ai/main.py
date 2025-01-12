import logging
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv
from langchain_community.document_loaders import PolarsDataFrameLoader

from planning_ai.common.utils import Paths
from planning_ai.graph import create_graph

load_dotenv()


def build_quarto_doc(doc_title, out):
    final = out["generate_final_summary"]

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
        "# Figures\n\n"
        "Figure @fig-wards shows the percentage of responses by total population"
        " within each Ward that had at least one response.\n\n"
        f"![Ward Proportions](./figs/wards.png){{#fig-wards}}\n\n"
        "Figure @fig-imd shows the percentage of responses by total population"
        " within each IMD quintile.\n\n"
        f"![IMD Quintile Props](./figs/imd_decile.png){{#fig-imd}}\n\n"
        "# Themes and Policies\n\n"
        "## Support\n\n"
        f"{final['policies_support']}"
        "## Object\n\n"
        f"{final['policies_object']}"
    )

    with open(Paths.SUMMARY / f"{doc_title.replace(' ', '_')}.qmd", "w") as f:
        f.write(quarto_doc)


def read_docs():
    df = pl.read_parquet(Paths.STAGING / "gcpt3.parquet")
    df = (
        df.filter(
            pl.col("representations_document") == "Local Plan Issues and Options Report"
        )
        .unique("id")
        .with_row_index()
    )
    loader = PolarsDataFrameLoader(df, page_content_column="text")

    docs = list(
        {
            doc.page_content: {"document": doc, "filename": doc.metadata["id"]}
            for doc in loader.load()
            if doc.page_content and len(doc.page_content.split(" ")) > 25
        }.values()
    )
    return docs


def process_postcodes(documents):
    postcodes = [doc["document"].metadata["respondentpostcode"] for doc in documents]
    postcodes = (
        pl.DataFrame({"postcode": postcodes})["postcode"]
        .value_counts()
        .with_columns(pl.col("postcode").str.replace_all(" ", ""))
    )
    onspd = pl.read_csv(
        "./data/raw/onspd/ONSPD_FEB_2024.csv", columns=["PCD", "OSWARD", "LSOA11"]
    ).with_columns(pl.col("PCD").str.replace_all(" ", "").alias("postcode"))
    postcodes = postcodes.join(onspd, on="postcode")
    return postcodes


def wards_pop(postcodes):
    wards = (
        pl.read_csv("./data/raw/TS001-2021-3-filtered-2025-01-09T11_07_15Z.csv")
        .with_columns(pl.col("Electoral wards and divisions Code").alias("OSWARD"))
        .group_by("OSWARD")
        .sum()
    )
    postcodes = postcodes.join(wards, on="OSWARD").with_columns(
        ((pl.col("count") / pl.col("Observation")) * 100).alias("prop")
    )
    ward_boundaries = gpd.read_file(
        "./data/raw/Wards_December_2021_GB_BFE_2022_7523259277605796091.zip"
    )
    ward_boundaries = ward_boundaries.merge(
        postcodes.to_pandas(), left_on="WD21CD", right_on="OSWARD"
    )

    _, ax = plt.subplots()
    ward_boundaries.plot(ax=ax, column="prop", legend=True)

    plt.axis("off")
    plt.savefig(Paths.SUMMARY / "figs" / "wards.png")


def imd_bar(postcodes):
    # Load the IMD data
    imd = pl.read_csv(
        "./data/raw/uk_imd2019.csv", columns=["LSOA", "LA_decile"]
    ).with_columns(((pl.col("LA_decile") - 1) // 2) + 1)
    pops = pl.read_excel(
        "./data/raw/sapelsoabroadage20112022.xlsx",
        sheet_name="Mid-2022 LSOA 2021",
        read_options={"header_row": 3},
        columns=["LSOA 2021 Code", "Total"],
    )

    # Join the postcodes data with IMD decile data
    postcodes = (
        postcodes.join(imd, left_on="LSOA11", right_on="LSOA")
        .join(pops, left_on="LSOA11", right_on="LSOA 2021 Code")
        .group_by("LA_decile")
        .agg(pl.col("count").sum(), pl.col("LSOA11").count(), pl.col("Total").sum())
        .sort("LA_decile")
        .with_columns(((pl.col("count") / pl.col("Total")) * 100).alias("prop"))
    )

    # Convert the Polars DataFrame to a Pandas DataFrame for plotting
    postcodes_pd = postcodes.to_pandas()

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots()

    # Plot the number of responses
    ax1.bar(
        postcodes_pd["LA_decile"],
        postcodes_pd["prop"],
        label="Percentage of Population (%)",
    )
    ax1.set_xlabel("IMD Quintile")
    ax1.set_ylabel("Proporition of Population (%)")
    ax1.tick_params(axis="y")

    plt.title("Comparison of Responses by IMD Decile")

    # Save the figure
    plt.tight_layout()
    plt.savefig(Paths.SUMMARY / "figs" / "imd_decile.png")
    # plt.show()


def main():
    docs = read_docs()[:500]
    n_docs = len(docs)

    logging.warning(f"{n_docs} documents being processed!")

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=10_240, chunk_overlap=0
    # )
    # split_docs = text_splitter.split_documents(docs)

    app = create_graph()

    step = None
    for step in app.stream({"documents": docs, "n_docs": n_docs}):
        print(step.keys())

    if step is None:
        raise ValueError("No steps were processed!")
    return step


if __name__ == "__main__":
    doc_title = "Cambridge Response Summary"

    tic = time.time()
    out = main()
    postcodes = process_postcodes(out["generate_final_summary"]["documents"])
    wards_pop(postcodes)
    imd_bar(postcodes)
    build_quarto_doc(doc_title, out)
    toc = time.time()

    print(f"Time taken: {(toc - tic) / 60:.2f} minutes.")
