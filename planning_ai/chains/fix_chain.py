from langchain_core.prompts import ChatPromptTemplate

from planning_ai.chains.map_chain import SLLM

with open("./planning_ai/chains/prompts/fix_hallucination.txt", "r") as f:
    map_template = f.read()

map_prompt = ChatPromptTemplate.from_messages([("system", map_template)])
fix_chain = map_prompt | SLLM

if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """

    result = fix_chain.invoke(
        {
            "summary": "This plan is great because they are building a nuclear power plant.",
            "explanation": "The original response does not mention a nuclear power plant, and appears to view the plan negatively",
            "context": test_document,
        }
    )
    print(result)
