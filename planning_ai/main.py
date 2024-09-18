import os
import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from opencage.geocoder import OpenCageGeocode

from planning_ai.common.utils import Paths
from planning_ai.graph import create_graph

load_dotenv()


def _geocode_points(x):
    api_key = os.getenv("OPENCAGE_API_KEY")
    geocoder = OpenCageGeocode(key=api_key)
    out = geocoder.geocode(x)
    if out:
        return out[0]["geometry"]
    else:
        return {"lat": -99.0, "lng": -99.0}


def map_locations(places_df: pl.DataFrame):
    lad = gpd.read_file(Paths.RAW / "LAD_BUC_2022.gpkg").to_crs("epsg:4326")
    lad_camb = lad[lad["LAD22NM"].str.contains("Cambridge")]
    places_df = places_df.with_columns(
        pl.col("Place")
        .map_elements(
            lambda x: _geocode_points(x),
            return_dtype=pl.Struct,
        )
        .alias("geometry")
    ).with_columns(pl.col("geometry").struct[0], pl.col("geometry").struct[1])

    places_pd = places_df.to_pandas()
    places_gdf = (
        gpd.GeoDataFrame(
            places_pd,
            geometry=gpd.points_from_xy(x=places_df["lng"], y=places_df["lat"]),
        )
        .set_crs("epsg:4326")
        .clip(lad)
    )

    _, ax = plt.subplots()
    lad.plot(ax=ax, color="white", edgecolor="gray")
    lad_camb.plot(ax=ax, color="white", edgecolor="black")
    places_gdf.plot(ax=ax, column="Mean Sentiment", markersize=5, legend=True)

    bounds = lad_camb.total_bounds
    buffer = 0.1
    ax.set_xlim([bounds[0] - buffer, bounds[2] + buffer])
    ax.set_ylim([bounds[1] - buffer, bounds[3] + buffer])
    plt.axis("off")
    plt.savefig(Paths.SUMMARY / "figs" / "places.png")


def build_quarto_doc(doc_title, out):
    final = out["generate_final_summary"]
    executive_summary = (
        final["final_summary"].split("## Key points raised in support")[0].strip()
    )
    key_points = final["final_summary"].split("## Key points raised in support")[1]

    aims = []
    for summary in final["summaries_fixed"]:
        aim = summary["summary"].aims
        aims.extend(aim)

    value_counts = Counter(aims)
    total_values = sum(value_counts.values())
    percentages = {
        key: {"count": count, "percentage": (count / total_values)}
        for key, count in value_counts.items()
    }
    top_5 = sorted(percentages.items(), key=lambda x: x[1]["percentage"], reverse=True)[
        :5
    ]
    thematic_breakdown = "| **Aim** | **Percentage** | **Count** |\n|---|---|---|\n"
    thematic_breakdown += "\n".join(
        [f"| {item} | {d['percentage']:.2%} | {d['count']} |" for item, d in top_5]
    )

    places_df = (
        pl.DataFrame(
            [
                place.dict()
                for summary in final["summaries_fixed"]
                for place in summary["summary"].places
            ]
        )
        .group_by("place")
        .agg(
            pl.col("place").len().alias("Count"),
            pl.col("sentiment").mean().alias("Mean Sentiment"),
        )
        .rename({"place": "Place"})
    )

    map_locations(places_df)

    places_breakdown = (
        places_df.sort("Count", descending=True)
        .head()
        .to_pandas()
        .to_markdown(index=False)
    )

    stances = [summary["summary"].stance for summary in final["summaries_fixed"]]
    value_counts = Counter(stances)
    total_values = sum(value_counts.values())
    percentages = {
        key: {"count": count, "percentage": (count / total_values)}
        for key, count in value_counts.items()
    }
    stances_top = sorted(
        percentages.items(), key=lambda x: x[1]["percentage"], reverse=True
    )
    stances_breakdown = " | ".join(
        [
            f"**{item}**: {stance['percentage']:.2%} _({stance['count']})_"
            for item, stance in stances_top
        ]
    )

    short_summaries = "\n\n".join(
        [
            f"#### **TODO**\n"
            f"{summary['summary'].summary}\n\n"
            f"**Stance**: {summary['summary'].stance}\n\n"
            f"**Constructiveness**: {summary['summary'].rating}\n\n"
            for summary in final["summaries_fixed"]
        ]
    )

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
        f"{executive_summary}\n\n"
        f"{stances_breakdown}\n\n"
        "## Aim Breakdown\n\n"
        "The aim breakdown identifies which aims are mentioned "
        "within each response. "
        "A single response may discuss multiple topics.\n"
        f"\n\n{thematic_breakdown}\n\n"
        f"\n\n{places_breakdown}\n\n"
        f"![Locations mentioned by sentiment](./figs/places.png)\n\n"
        "## Key points raised in support\n\n"
        f"{key_points}\n\n"
        "## Summaries\n"
        f"{short_summaries}"
    )

    with open(Paths.SUMMARY / f"{doc_title.replace(' ', '_')}.qmd", "w") as f:
        f.write(quarto_doc)


def main():
    loader = DirectoryLoader(
        path=str(Paths.STAGING),
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        recursive=True,
    )
    docs = [doc for doc in loader.load() if doc.page_content]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    app = create_graph()

    step = None
    for step in app.stream(
        {
            "documents": [doc.page_content for doc in split_docs],
            "filenames": [Path(doc.metadata["source"]) for doc in split_docs],
        }
    ):
        print(list(step.keys()))

    if step is None:
        raise ValueError("No steps were processed!")

    return step


if __name__ == "__main__":
    doc_title = "Cambridge Response Summary"

    tic = time.time()
    out = main()
    build_quarto_doc(doc_title, out)
    toc = time.time()

    print(f"Time taken: {(toc - tic) / 60:.2f} minutes.")
