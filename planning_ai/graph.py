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
from planning_ai.states import OverallState


def handle_hallucination_cycle(state: OverallState):
    if any(h["hallucination"].score > 0 for h in state["hallucinations"]):
        return [
            Send("check_hallucination", {"document": doc})
            for doc in state["documents"]
        ]
    else:
        return [Send("collect_summaries", state)]

def create_graph():
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("fix_hallucination", fix_hallucination)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    graph.set_conditional_entry_point(map_summaries, ["generate_summary"])
    graph.add_conditional_edges(
        "generate_summary", map_hallucinations, ["check_hallucination"]
    )
    graph.add_conditional_edges(
        "check_hallucination", map_fix_hallucinations, ["fix_hallucination"]
    )
    graph.add_edge("collect_summaries", "generate_final_summary")
    graph.add_edge("generate_final_summary", END)

    return graph.compile()
