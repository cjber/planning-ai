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
