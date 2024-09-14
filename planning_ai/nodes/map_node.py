import logging

logging.basicConfig(level=logging.WARNING)

from langchain_core.documents import Document
from langgraph.constants import Send

from planning_ai.chains.map_chain import map_chain
from planning_ai.states import DocumentState, OverallState


def generate_summary(state: DocumentState):
    response = map_chain.invoke({"context": state["document"]})
    return {
        "summaries": [
            {
                "summary": response,
                "document": state["document"],
                "filename": state["filename"],
                "iteration": 1,
            }
        ]
    }


def map_summaries(state: OverallState):
    return [
        Send(
            "generate_summary",
            {"document": document, "filename": filename},
        )
        for document, filename in zip(state["documents"], state["filenames"])
    ]


def collect_summaries(state: OverallState):
    print("test")
    __import__("ipdb").set_trace()
    state.keys()
    len(state["documents"])
    len(state["summaries_fixed"])
    len(state["hallucinations"])
    state["hallucinations"]
    return {
        "summary_documents": [
            Document(
                page_content=hallucination.summary,
                metadata={
                    "stance": hallucination.stance,
                    "aims": hallucination.aims,
                    "places": hallucination.places,
                    "rating": hallucination.rating,
                    "hallucination": hallucination.score,
                    "explanation": hallucination.explanation,
                },
            )
        ]
        for hallucination in state["summaries_fixed"]
    }
