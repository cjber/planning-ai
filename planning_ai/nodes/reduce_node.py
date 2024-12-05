from planning_ai.chains.reduce_chain import reduce_chain
from planning_ai.states import OverallState


def generate_final_summary(state: OverallState):
    """Generates a final summary from fixed summaries.

    This function checks if the number of documents matches the number of fixed summaries.
    It then filters the summaries to include only those with a non-neutral stance and a
    rating of 5 or higher (constructiveness). These filtered summaries are then combined
    into a final summary using the `reduce_chain`.

    Args:
        state (OverallState): The overall state containing documents, summaries, and
            other related information.

    Returns:
        dict: A dictionary containing the final summary, along with the original
        documents, summaries, fixed summaries, and hallucinations.
    """
    if len(state["documents"]) == len(state["summaries_fixed"]):
        summaries = [
            str(summary["summary"])
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
