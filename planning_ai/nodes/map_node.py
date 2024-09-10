from langchain_core.documents import Document
from langgraph.constants import Send

from planning_ai.chains.map_chain import map_chain
from planning_ai.states import OverallState, SummaryState


def generate_summary(state: SummaryState):
    response = map_chain.invoke(
        {"context": state["content"], "filename": state["filename"]}
    )
    return {
        "summaries": [
            {
                "response": response,
                "original": state["content"],
                "filename": state["filename"],
            }
        ]
    }


def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content, "filename": filename})
        for content, filename in zip(state["contents"], state["filenames"])
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [
            Document(
                page_content=summary["response"].summary,
                metadata={
                    "stance": summary["response"].stance,
                    "themes": summary["response"].themes,
                    "places": summary["response"].places,
                    "rating": summary["response"].rating,
                    "original": summary["original"],
                    "filename": summary["filename"],
                    "index": idx,
                },
            )
            for idx, summary in enumerate(state["summaries"], start=1)
        ]
    }
