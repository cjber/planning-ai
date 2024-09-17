from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.states import OverallState


def generate_final_summary(state: OverallState):
    if len(state["documents"]) == len(state["summaries_fixed"]):
        summaries = [
            summary["summary"].summary
            for summary in state["summaries_fixed"]
            if summary["summary"].stance != "NEUTRAL" and summary["summary"].rating >= 5
        ]
        response = reduce_chain.invoke({"context": summaries})
        return {
            "final_summary": response,
            "summaries_fixed": state["summaries_fixed"],
            "summaries": state["summaries"],
            "hallucinations": state["hallucinations"],
            "documents": state["documents"],
        }


def add_snippets(state: OverallState):
    final_summary = state["final_summary"]
    summaries = state["summaries_fixed"]

    response = snippet_chain.invoke(
        {"final_summary": final_summary, "summaries": summaries}
    )
