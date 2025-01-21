import json
import logging
from pathlib import Path

import polars as pl

from planning_ai.chains.policy_chain import policy_chain
from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.logging import logger
from planning_ai.states import OverallState
from planning_ai.themes import THEMES_AND_POLICIES


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
            "policies": doc["policies"],
            "notes": doc["notes"],
            "refinement_attempts": doc["refinement_attempts"],
            "hallucination": doc["hallucination"].model_dump(),
            "is_hallucinated": doc["is_hallucinated"],
            "failed": doc["failed"],
        }
        for doc in docs
    ]
    for doc in out:
        filename = Path(str(doc["filename"])).stem
        with open(f"data/out/summaries/{filename}.json", "w") as f:
            json.dump(doc, f)


def extract_policies_from_docs(docs):
    policies = {"doc_id": [], "themes": [], "policies": [], "details": [], "stance": []}
    for doc in docs:
        if not doc["summary"].policies:
            continue
        for policy in doc["summary"].policies:
            for theme, p in THEMES_AND_POLICIES.items():
                if policy.policy in p:
                    policies["doc_id"].append(doc["doc_id"])
                    policies["themes"].append(theme)
                    policies["policies"].append(policy.policy)
                    policies["details"].append(policy.note)
                    policies["stance"].append(
                        doc["document"].metadata["representations_support/object"]
                    )
    return pl.DataFrame(policies)


def add_doc_id(final_docs):
    out_docs = []
    for id, doc in enumerate(final_docs):
        doc["summary"].summary = f"Document ID: [{id}]\n\n{doc['summary'].summary}"
        doc["doc_id"] = id
        out_docs.append(doc)
    return out_docs


def batch_generate_executive_summaries(summaries):
    """Processes summaries to generate final responses.

    Args:
        summaries (list): A list of summary dictionaries.

    Returns:
        list: A list of final responses.
    """
    summaries_text = [
        f"Document ID: {[s['doc_id']]} {s['summary'].summary}" for s in summaries
    ]
    final_responses = []
    batch_size = 50
    for i in range(0, len(summaries_text), batch_size):
        logger.info(
            f"Processing batches... {int(i/50)+1}/{(len(summaries_text)//batch_size)+1}"
        )
        batch = summaries_text[i : i + batch_size]
        response = reduce_chain.invoke({"context": batch})
        final_responses.append(response)
    return final_responses


def generate_policy_output(policy_groups):
    out = []
    for policy in (
        policy_groups.group_by(["themes", "policies", "stance"])
        .agg(pl.col("details"), pl.col("doc_id"))
        .rows(named=True)
    ):
        logger.info(f"Processing policies: {policy['policies']}...")
        zipped = [
            f"{bullet} Doc ID: {id}"
            for (bullet, id) in zip(policy["details"], policy["doc_id"], strict=True)
        ]
        reduced = policy_chain.invoke(
            {
                "theme": policy["themes"],
                "policy": policy["policies"],
                "details": zipped,
            }
        )
        out.extend(policy | p for p in reduced.dict()["policies"])
    return (
        pl.DataFrame(out)
        .group_by(["themes", "policies", "stance"])
        .agg(["detail", "doc_id"])
    )


def generate_final_report(state: OverallState):
    final_docs = [doc for doc in state["documents"] if doc["processed"]]
    if len(final_docs) == state["n_docs"]:
        logger.info(f"Generating final report... ({len(final_docs)} documents)")
        return final_output(final_docs)


def final_output(final_docs):
    docs = [doc for doc in final_docs if not doc["failed"]]

    docs = add_doc_id(docs)

    policy_groups = extract_policies_from_docs(docs)
    policies = generate_policy_output(policy_groups)

    batch_executive = batch_generate_executive_summaries(docs)
    executive = reduce_chain.invoke({"context": "\n\n".join(batch_executive)})
    return {"executive": executive, "documents": docs, "policies": policies}
