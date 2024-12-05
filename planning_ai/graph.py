from langgraph.constants import START
from langgraph.graph import END, StateGraph

from planning_ai.nodes.hallucination_node import (
    check_hallucination,
    fix_hallucination,
    map_fix_hallucinations,
    map_hallucinations,
)
from planning_ai.nodes.map_node import generate_summary, map_summaries
from planning_ai.nodes.reduce_node import generate_final_summary
from planning_ai.states import OverallState


def create_graph():
    """Creates and compiles a state graph for document processing.

    This function sets up a state graph using the `StateGraph` class, defining nodes
    and edges for processing documents. It includes nodes for generating summaries,
    checking and fixing hallucinations, and generating a final summary. Conditional
    edges are added to manage the flow of data between nodes based on the processing
    state.

    Returns:
        StateGraph: The compiled state graph ready for execution.
    """
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("fix_hallucination", fix_hallucination)
    # graph.add_node("generate_final_summary", generate_final_summary)

    graph.add_conditional_edges(
        START,
        map_summaries,
        ["generate_summary"],
    )
    graph.add_conditional_edges(
        "generate_summary",
        map_hallucinations,
        ["check_hallucination"],
    )
    graph.add_conditional_edges(
        "check_hallucination",
        map_fix_hallucinations,
        ["fix_hallucination"],
    )
    graph.add_conditional_edges(
        "fix_hallucination",
        map_hallucinations,
        ["check_hallucination"],
    )

    # graph.add_edge("check_hallucination", "generate_final_summary")
    # graph.add_edge("generate_final_summary", END)

    return graph.compile()
