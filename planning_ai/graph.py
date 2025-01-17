from langgraph.constants import START
from langgraph.graph import END, StateGraph

from planning_ai.nodes.hallucination_node import (
    check_hallucination,
    fix_hallucination,
    map_check,
    map_fix,
)
from planning_ai.nodes.map_node import (
    add_entities,
    generate_summary,
    map_documents,
    retrieve_themes,
)
from planning_ai.nodes.reduce_node import generate_final_report
from planning_ai.states import OverallState


def create_graph():
    graph = StateGraph(OverallState)
    graph.add_node("add_entities", add_entities)
    graph.add_node("retrieve_themes", retrieve_themes)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("fix_hallucination", fix_hallucination)
    graph.add_node("generate_final_report", generate_final_report)

    graph.add_edge(START, "add_entities")
    graph.add_conditional_edges("add_entities", map_documents, ["generate_summary"])
    graph.add_conditional_edges("generate_summary", map_check, ["check_hallucination"])
    graph.add_conditional_edges("check_hallucination", map_fix, ["fix_hallucination"])
    graph.add_conditional_edges("fix_hallucination", map_check, ["check_hallucination"])

    graph.add_edge("check_hallucination", "generate_final_report")
    graph.add_edge("generate_final_report", END)

    return graph.compile()


def plot_mermaid():
    graph = create_graph()
    print(graph.get_graph().draw_mermaid())
