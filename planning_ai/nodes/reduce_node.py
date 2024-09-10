from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.states import OverallState


def generate_final_summary(state: OverallState):
    response = reduce_chain.invoke({"context": state["collapsed_summaries"]})
    return {
        "final_summary": response,
        "collapsed_summaries": state["collapsed_summaries"],
    }
