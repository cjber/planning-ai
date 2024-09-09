from langgraph.graph import END, START, StateGraph

from planning_ai.nodes.map_node import (
    collect_summaries,
    generate_summary,
    map_summaries,
    should_collapse,
)
from planning_ai.nodes.reduce_node import collapse_summaries, generate_final_summary
from planning_ai.states import OverallState


def create_graph():
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    return graph.compile()
