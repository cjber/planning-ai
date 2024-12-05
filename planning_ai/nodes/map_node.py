import json
from pathlib import Path

from langgraph.constants import Send
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from planning_ai.chains.map_chain import map_chain
from planning_ai.common.utils import Paths
from planning_ai.states import DocumentState, OverallState

anonymizer = AnonymizerEngine()
analyzer = AnalyzerEngine()


def remove_pii(document: str) -> str:
    """Removes personally identifiable information (PII) from a document.

    This function uses the Presidio Analyzer and Anonymizer to detect and anonymize
    PII such as names, phone numbers, and email addresses in the given document.

    Args:
        document (str): The document text from which PII should be removed.

    Returns:
        str: The document text with PII anonymized.
    """
    results = analyzer.analyze(
        text=document,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        language="en",
    )
    document = anonymizer.anonymize(text=document, analyzer_results=results)
    return document


def generate_summary(state: DocumentState) -> dict:
    """Generates a summary for a document after removing PII.

    This function first anonymizes the document to remove PII, then generates a summary
    using the `map_chain`. The summary is added to the document state.

    Args:
        state (DocumentState): The current state of the document, including its text
            and filename.

    Returns:
        dict: A dictionary containing the generated summary and updated document state.
    """
    state["document"] = remove_pii(state["document"])
    response = map_chain.invoke({"context": state["document"]})
    summary = response.summary
    themes = [theme.value for theme in response.themes]
    policies = [policy.dict() for policy in response.policies]

    out_policies = []
    for theme in policies:
        name = theme["theme"].value
        policy_list = theme["policies"]
        out_policies.append({"theme": name, "policies": policy_list})

    out_places = []
    for place in response.places:
        name = place.place
        sentiment = place.sentiment.value
        out_places.append({"place": name, "sentiment": sentiment})

    save_output = {
        "summary": summary,
        "themes": themes,
        "policies": out_policies,
        "places": out_places,
    }

    outfile = f"{Path(state["filename"]).stem}_summary.json"
    with open(Paths.SUMMARIES / outfile, "w") as file:
        json.dump(save_output, file, indent=4)

    output = {
        "summary": response,
        "document": state["document"],
        "filename": str(state["filename"]),
        "iteration": 1,
    }

    return {"summaries": [output]}


def map_summaries(state: OverallState) -> list[Send]:
    """Maps documents to the `generate_summary` function for processing.

    This function prepares a list of documents to be summarized by sending them to the
    `generate_summary` function. It allows for parallel processing of document summaries.

    Args:
        state (OverallState): The overall state containing all documents and their filenames.

    Returns:
        list: A list of Send objects directing each document to the `generate_summary`
        function.
    """
    return [
        Send(
            "generate_summary",
            {"document": document, "filename": filename},
        )
        for document, filename in zip(state["documents"], state["filenames"])
    ]
