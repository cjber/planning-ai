from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.states import OverallState


def generate_final_summary(state: OverallState):
    __import__("ipdb").set_trace()
    # response = reduce_chain.invoke({"context": state["summary_documents"]})
    return {
        # "final_summary": response,
        "summaries": state["summary_documents"],
        "hallucinations": state["hallucinations"],
        "summary": state["summary_documents"],
    }
