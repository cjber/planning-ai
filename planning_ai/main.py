from collections import Counter
from pathlib import Path

import polars as pl
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

from planning_ai.common.utils import Paths
from planning_ai.graph import create_graph


def build_quarto_doc(doc_title, out):
    final = out["generate_final_summary"]
    executive_summary = (
        final["final_summary"].split("## Key points raised in support")[0].strip()
    )
    key_points = final["final_summary"].split("## Key points raised in support")[1]

    aims = [
        aim
        for summary in final["collapsed_summaries"]
        for aim in summary.metadata["aims"]
    ]
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
                for summary in final["collapsed_summaries"]
                for place in summary.metadata["places"]
            ]
        )
        .group_by("place")
        .agg(
            pl.col("place").len().alias("Count"),
            pl.col("sentiment").mean().alias("Mean Sentiment"),
        )
        .rename({"place": "Place"})
    )
    places_breakdown = (
        places_df.sort("Count", descending=True)
        .head()
        .to_pandas()
        .to_markdown(index=False)
    )

    # places_breakdown = "| **Place** | **Percentage** | **Count** |\n|---|---|---|\n"
    # places_breakdown += "\n".join(
    #     [f"| {item} | {d['percentage']:.2%} | {d['count']} |" for item, d in top_5]
    # )

    stances = [summary.metadata["stance"] for summary in final["collapsed_summaries"]]
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
            f"#### {summary.metadata['filename']}\n"
            f"{summary.page_content}\n\n"
            f"**Stance**: {summary.metadata['stance']}\n\n"
            f"**Constructiveness**: {summary.metadata['rating']}\n\n"
            for summary in final["collapsed_summaries"]
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
        "## Key points raised in support\n\n"
        f"{key_points}\n\n"
        "## Summaries\n"
        f"{short_summaries}"
    )

    with open(f"./reports/{doc_title.replace(' ', '_')}.qmd", "w") as f:
        f.write(quarto_doc)


def main():
    loader = DirectoryLoader(
        path=str(Paths.STAGING),
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        recursive=True,
    )
    docs = [doc for doc in loader.load()[:20] if doc.page_content]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    app = create_graph()

    step = None
    for step in app.stream(
        {
            "contents": [doc.page_content for doc in split_docs],
            "filenames": [Path(doc.metadata["source"]) for doc in split_docs],
        },
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))

    if step is None:
        raise ValueError("No steps were processed!")

    return step


if __name__ == "__main__":
    doc_title = "Cambridge Response Summary"
    out = main()
    build_quarto_doc(doc_title, out)
