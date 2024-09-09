from langchain.chains.combine_documents.reduce import collapse_docs, split_list_of_docs

from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.common.utils import Consts, length_function
from planning_ai.states import OverallState


def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, Consts.TOKEN_MAX
    )
    results = []
    for doc_list in doc_lists:
        results.append(collapse_docs(doc_list, reduce_chain.invoke))

    return {"collapsed_summaries": results}


def generate_final_summary(state: OverallState):
    response = reduce_chain.invoke({"context": state["collapsed_summaries"]})
    return {
        "final_summary": response,
        "collapsed_summaries": state["collapsed_summaries"],
    }
