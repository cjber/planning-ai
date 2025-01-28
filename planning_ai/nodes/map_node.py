import spacy
from langgraph.types import Send
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from planning_ai.chains.map_chain import create_dynamic_map_chain, map_template
from planning_ai.chains.themes_chain import themes_chain
from planning_ai.logging import logger
from planning_ai.states import DocumentState, OverallState

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

nlp = spacy.load("en_core_web_lg")



def retrieve_themes(state: DocumentState) -> DocumentState:
    try:
        result = themes_chain.invoke({"document": state["document"].page_content})
        if not result.themes:
            state["themes"] = set()
            return state
        themes = [theme.value for theme in result.themes]
    except Exception as e:
        logger.error(f"Theme selection error: {e}")
        themes = []

    state["themes"] = set(themes)
    return state


def add_entities(state: OverallState) -> OverallState:
    logger.info("Adding entities to all documents.")
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
    results = analyzer.analyze(
        text=document,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        language="en",
    )
    return anonymizer.anonymize(text=document, analyzer_results=results).text


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
    logger.info(f"Generating summary for document: {state['filename']}")

    logger.info(f"Starting PII removal for: {state['filename']}")
    state["document"].page_content = remove_pii(state["document"].page_content)
    logger.info(f"Retrieving themes for: {state['filename']}")
    state = retrieve_themes(state)

    if not state["themes"]:
        logger.warning(f"No themes found for {state['filename']}")
        return {
            "documents": [
                {
                    **state,
                    "summary": "",
                    "refinement_attempts": 0,
                    "is_hallucinated": True,
                    "processed": True,
                    "failed": True,
                }
            ]
        }

    map_chain = create_dynamic_map_chain(themes=state["themes"], prompt=map_template)
    try:
        response = map_chain.invoke({"context": state["document"].page_content})
    except Exception as e:
        logger.error(f"Failed to decode JSON {state['document']}: {e}")
        return {
            "documents": [
                {
                    **state,
                    "summary": "",
                    "refinement_attempts": 0,
                    "is_hallucinated": True,
                    "processed": True,
                    "failed": True,
                }
            ]
        }
    logger.info(f"Summary generation completed for document: {state['filename']}")

    return {
        "documents": [
            {
                **state,
                "summary": response,
                "refinement_attempts": 0,
                "is_hallucinated": True,  # start true to ensure cycle begins
                "failed": False,
                "processed": False,
            }
        ]
    }


def map_documents(state: OverallState) -> list[Send]:
    logger.info("Mapping documents to generate summaries.")
    return [Send("generate_summary", document) for document in state["documents"]]
