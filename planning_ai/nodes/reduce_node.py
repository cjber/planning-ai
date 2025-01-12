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


def extract_policies_from_docs(docs):
    policies = {"themes": [], "policies": [], "details": [], "stance": []}
    for doc in docs:
        if not doc["summary"].policies:
            continue
        for policy in doc["summary"].policies:
            for theme, p in THEMES_AND_POLICIES.items():
                if policy.policy.name in p:
                    policies["themes"].append(theme)
                    policies["policies"].append(policy.policy.name)
                    policies["details"].append(
                        f"{policy.note} [{doc['document'].metadata['index']}]"
                    )
                    policies["stance"].append(
                        doc["document"].metadata["representations_support/object"]
                    )
    df = pl.DataFrame(policies)
    grouped = df.group_by(["themes", "policies", "stance"]).agg(pl.col("details"))
    return grouped


def filter_final_documents(state: OverallState):
    return [doc for doc in state["documents"] if doc["hallucination"].score == 1]


def filter_docs(final_docs):
    out_docs = []
    for doc in final_docs:
        if (
            (doc["summary"].summary != "INVALID")
            and (doc["themes"] != set())
            and (doc["iteration"] != 99)
        ):
            doc["summary"].summary = (
                f"Document ID: [{doc['document'].metadata['index']}]\n\n{doc['summary'].summary}"
            )
            out_docs.append(doc)
    return out_docs


def save_summaries_to_json(docs):
    """Saves summaries to JSON files.

    Args:
        out (list): A list of summary dictionaries.
    """
    out = [
        {
            "document": doc["document"].model_dump()["page_content"],
            **doc["document"].metadata,
            "filename": doc["filename"],
            "entities": doc["entities"],
            "themes": list(doc["themes"]),
            "summary": doc["summary"].model_dump()["summary"],
            "policies": [
                {"policy": policy["policy"].name, "note": policy["note"]}
                for policy in (doc["summary"].model_dump().get("policies", []) or [])
            ],
            "iteration": doc["iteration"],
            "hallucination": doc["hallucination"].model_dump(),
        }
        for doc in docs
    ]
    for doc in out:
        filename = Path(str(doc["filename"])).stem
        with open(f"data/out/summaries/{filename}.json", "w") as f:
            json.dump(doc, f)


def batch_generate_executive_summaries(summaries):
    """Processes summaries to generate final responses.

    Args:
        summaries (list): A list of summary dictionaries.

    Returns:
        list: A list of final responses.
    """
    summaries_text = [
        f"Document ID: {[s['document'].metadata['index']]} {s['summary'].summary}"
        for s in summaries
    ]
    final_responses = []
    batch_size = 50
    for i in range(0, len(summaries_text), batch_size):
        logger.warning(
            f"Processing batches... {i/50}/{len(summaries_text)//batch_size}"
        )
        batch = summaries_text[i : i + batch_size]
        response = reduce_chain.invoke({"context": batch})
        final_responses.append(response)
    return final_responses


def generate_policy_output(policy_groups):
    policies_support = []
    policies_object = []
    for _, policy in policy_groups.group_by(["themes", "policies"]):
        logger.warning("Processing policies.")
        bullets = "* " + "* \n".join(policy["details"][0])
        pchain_out = policy_chain.invoke(
            {"policy": policy["policies"][0], "bullet_points": bullets}
        )
        if policy["stance"][0] == "Support":
            policies_support.append(
                {
                    "theme": policy["themes"][0],
                    "policy": policy["policies"][0],
                    "points": pchain_out,
                }
            )
        else:
            policies_object.append(
                {
                    "theme": policy["themes"][0],
                    "policy": policy["policies"][0],
                    "points": pchain_out,
                }
            )
    return policies_support, policies_object


def format_themes(policies):
    themes = ""
    for theme, policies in pl.DataFrame(policies).group_by("theme"):
        themes += f"### {theme[0]}\n\n"
        for row in policies.iter_rows(named=True):
            themes += f"\n#### {row['policy']}\n\n"
            themes += f"{row['points']}\n"
        themes += "\n"
    return themes


def generate_final_report(state: OverallState):
    logger.warning("Generating final summary")
    final_docs = filter_final_documents(state)
    logger.warning(f"Number of final docs: {len(final_docs)}")

    if len(final_docs) == state["n_docs"]:
        docs = filter_docs(final_docs)
        save_summaries_to_json(docs)

        policy_groups = extract_policies_from_docs(docs)
        policies_support, policies_object = generate_policy_output(policy_groups)

        batch_executive = batch_generate_executive_summaries(docs)
        executive = reduce_chain.invoke({"context": "\n\n".join(batch_executive)})

        return {
            "executive": executive,
            "documents": final_docs,
            "policies_support": format_themes(policies_support),
            "policies_object": format_themes(policies_object),
        }
