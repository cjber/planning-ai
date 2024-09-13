from langgraph.constants import START, Send
from langgraph.graph import END, StateGraph

from planning_ai.nodes.hallucination_node import (
    check_hallucination,
    fix_hallucination,
    map_fix_hallucinations,
    map_hallucinations,
)
from planning_ai.nodes.map_node import (
    collect_summaries,
    generate_summary,
    map_summaries,
)
from planning_ai.nodes.reduce_node import generate_final_summary
from planning_ai.states import DocumentState, OverallState


def create_graph():
    subgraph = StateGraph(DocumentState)
    subgraph.add_node("generate_summary", generate_summary)
    # subgraph.add_node("check_hallucination", check_hallucination)
    # subgraph.add_node("fix_hallucination", fix_hallucination)

    subgraph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    # subgraph.add_conditional_edges(
    #     "generate_summary", map_hallucinations, ["check_hallucination"]
    # )
    # subgraph.add_conditional_edges(
    #     "check_hallucination", map_fix_hallucinations, ["fix_hallucination"]
    # )
    # subgraph.add_conditional_edges(
    #     "fix_hallucination", map_hallucinations, ["check_hallucination"]
    # )
    subgraph = subgraph.compile()

    graph = StateGraph(OverallState)
    graph.add_node("summary_graph", subgraph)
    # graph.add_node("collect_summaries", collect_summaries)

    graph.add_edge(START, "summary_graph")
    # graph.add_conditional_edges(
    #     "summary_graph", map_hallucinations, ["collect_summaries"]
    # )
    # graph.add_edge("generate_final_summary", END)

    return graph.compile()


# graph = create_graph()
# graph.get_graph().draw_png("test.png")
