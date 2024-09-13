import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

from langchain_core.documents import Document
from langgraph.constants import Send

from planning_ai.chains.map_chain import map_chain
from planning_ai.states import DocumentState, OverallState


def generate_summary(state: DocumentState):
    response = map_chain.invoke({"context": state["document"]})
    return {"summaries": [{"summary": response, "document": state["document"]}]}


def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"document": document})
        for document in state["documents"]
    ]


def collect_summaries(state: DocumentState):
    logging.debug(f"Collecting summary for document: {state['document']}")
    summary_document = Document(
        page_content=state["summary"].summary,
        metadata={
            "stance": state["summary"].stance,
            "aims": state["summary"].aims,
            "places": state["summary"].places,
            "rating": state["summary"].rating,
            "hallucination": state["hallucination"].score,
            "explanation": state["hallucination"].explanation,
        },
    )
    logging.debug(f"Summary document created: {summary_document}")
    return {
        "summary_documents": [summary_document]
        "summary_documents": [
            Document(
                page_content=state["summary"].summary,
                metadata={
                    "stance": state["summary"].stance,
                    "aims": state["summary"].aims,
                    "places": state["summary"].places,
                    "rating": state["summary"].rating,
                    "hallucination": state["hallucination"].score,
                    "explanation": state["hallucination"].explanation,
                },
            )
        ]
    }
