from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.states import OverallState


def generate_final_summary(state: OverallState):
    if len(state["documents"]) == len(state["summaries_fixed"]):
        response = reduce_chain.invoke({"context": state["summaries_fixed"]})
        return {
            "final_summary": response,
            "summaries_fixed": state["summaries_fixed"],
            "summaries": state["summary_documents"],
            "hallucinations": state["hallucinations"],
            "documents": state["documents"],
        }
