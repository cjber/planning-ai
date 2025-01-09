import json
import logging
from pathlib import Path

import polars as pl

from planning_ai.chains.policy_chain import policy_chain
from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.states import OverallState
from planning_ai.themes import THEMES_AND_POLICIES

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# TODO: Reduce down the grouping of policies. I.e. combine points that are closely related and add citations.
def markdown_bullets(summaries):
    policies = {"themes": [], "policies": [], "details": []}
    for summary in summaries:
        if not summary["summary"].policies:
            continue
        for policy in summary["summary"].policies:
            for theme, p in THEMES_AND_POLICIES.items():
                if policy.policy.name in p:
                    policies["themes"].append(theme)
                    policies["policies"].append(policy.policy.name)
                    policies["details"].append(policy.note)
    df = pl.DataFrame(policies)

    grouped = df.group_by(["themes", "policies"]).agg(pl.col("details"))
    return grouped


def generate_final_summary(state: OverallState):
    """Generates a final summary from fixed summaries.

    This function checks if the number of documents matches the number of fixed summaries.
    It then filters the summaries to include only those with a non-neutral stance and a
    rating of 5 or higher (constructiveness). These filtered summaries are then combined
    into a final summary using the `reduce_chain`.

    Args:
        state (OverallState): The overall state containing documents, summaries, and
            other related information.

    Returns:
        dict: A dictionary containing the final summary, along with the original
        documents, summaries, fixed summaries, and hallucinations.
    """
    logger.warning("Generating final summary")
    final_docs = [doc for doc in state["documents"] if doc["hallucination"].score == 1]
    logger.warning(f"Number of final docs: {len(final_docs)}")

    if len(final_docs) == state["n_docs"]:
        summaries = [
            # f"Document ID: [{id}]\n\n{doc["summary"].summary}"
            doc
            for id, doc in zip(range(state["n_docs"]), final_docs)
            if doc["summary"].summary != "INVALID"
            and doc["themes"] != set()
            and doc["iteration"] != 99
        ]

        out = []
        for doc in summaries:
            summary = doc["summary"].model_dump()
            if summary["policies"] is not None:
                policies = [
                    {"policy": policy["policy"].name, "note": policy["note"]}
                    for policy in summary["policies"]
                ]
            else:
                policies = []
            summary = summary["summary"]
            out.append(
                {
                    "document": doc["document"].model_dump()["page_content"],
                    "filename": doc["filename"],
                    "entities": doc["entities"],
                    "theme_docs": [d.model_dump() for d in doc["theme_docs"]],
                    "themes": list(doc["themes"]),
                    "summary": summary,
                    "policies": policies,
                    "iteration": doc["iteration"],
                    "hallucination": doc["hallucination"].model_dump(),
                }
            )

        for doc in out:
            filename = Path(str(doc["filename"])).stem
            with open(f"data/out/summaries/{filename}.json", "w") as f:
                json.dump(doc, f)

        summaries_text = [s["summary"].summary for s in summaries]
        final_responses = []
        batch_size = 50
        for i in range(0, len(summaries_text), batch_size):
            logger.warning("Processing batches.")
            batch = summaries_text[i : i + batch_size]
            response = reduce_chain.invoke({"context": batch})
            final_responses.append(response)

        final_response = reduce_chain.invoke({"context": "\n\n".join(final_responses)})
        pols = markdown_bullets(summaries)

        pol_out = []
        for _, policy in pols.group_by(["themes", "policies"]):
            logger.warning("Processing policies.")
            bullets = "* " + "* \n".join(policy["details"][0])
            pchain_out = policy_chain.invoke(
                {"policy": policy["policies"][0], "bullet_points": bullets}
            )
            pol_out.append(
                {
                    "theme": policy["themes"][0],
                    "policy": policy["policies"][0],
                    "points": pchain_out,
                }
            )

        themes = ""
        for theme, policies in pl.DataFrame(pol_out).group_by("theme"):
            themes += f"# {theme[0]}\n\n"
            for row in policies.iter_rows(named=True):
                themes += f"\n## {row['policy']}\n\n"
                themes += f"{row['points']}\n"
            themes += "\n"

        return {
            "final_summary": final_response,
            "documents": final_docs,
            "policies": themes,
        }
