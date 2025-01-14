import json
import logging

import spacy
from langchain_core.exceptions import OutputParserException
from langgraph.types import Send
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel

from planning_ai.chains.hallucination_chain import HallucinationChecker
from planning_ai.chains.map_chain import create_dynamic_map_chain, map_template
from planning_ai.chains.themes_chain import themes_chain
from planning_ai.states import DocumentState, OverallState

logging.basicConfig(
    level=logging.WARN, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BasicSummaryBroken(BaseModel):
    summary: str
    policies: None


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

nlp = spacy.load("en_core_web_lg")


def _return_summary_error(state):
    state["iteration"] = 99
    state["hallucination"] = HallucinationChecker(score=1, explanation="INVALID")
    state["summary"] = BasicSummaryBroken(summary="INVALID", policies=None)
    return {"documents": [state]}


def retrieve_themes(state: DocumentState) -> DocumentState:
    result = themes_chain.invoke({"document": state["document"]})
    if not result.themes:
        state["themes"] = set()
        return state
    themes = [theme.value for theme in result.themes]
    state["themes"] = set(themes)
    logger.warning(f"Retrieved relevant themes for: {state['filename']}")
    return state


def add_entities(state: OverallState) -> OverallState:
    for idx, document in enumerate(
        nlp.pipe(
            [doc["document"].page_content for doc in state["documents"]],
        )
    ):
        state["documents"][idx]["entities"] = [
            {"entity": ent.text, "label": ent.label_} for ent in document.ents
        ]
    return state


def remove_pii(document: str) -> str:
    """Removes personally identifiable information (PII) from a document.

    This function uses the Presidio Analyzer and Anonymizer to detect and anonymize
    PII such as names, phone numbers, and email addresses in the given document.

    Args:
        document (str): The document text from which PII should be removed.

    Returns:
        str: The document text with PII anonymized.
    """
    logger.warning("Starting PII removal.")
    results = analyzer.analyze(
        text=document,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        language="en",
    )
    document = anonymizer.anonymize(text=document, analyzer_results=results).text
    logger.warning("PII removal completed.")
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
    logger.warning(f"Generating summary for document: {state['filename']}")

    state["document"].page_content = remove_pii(state["document"].page_content)
    state = retrieve_themes(state)

    if not state["themes"]:
        return _return_summary_error(state)
    map_chain = create_dynamic_map_chain(themes=state["themes"], prompt=map_template)
    try:
        response = map_chain.invoke({"context": state["document"].page_content})
    except (OutputParserException, json.JSONDecodeError) as e:
        logger.error(f"Failed to decode JSON: {e}.")
        return _return_summary_error(state)
    logger.warning(f"Summary generation completed for document: {state['filename']}")
    return {"documents": [{**state, "summary": response, "iteration": 1}]}


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
    logger.warning("Mapping documents to generate summaries.")
    return [Send("generate_summary", document) for document in state["documents"]]
