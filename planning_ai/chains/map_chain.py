from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from planning_ai.llms.llm import LLM

with open("./planning_ai/chains/prompts/map.txt", "r") as f:
    map_template = f.read()

map_prompt = ChatPromptTemplate.from_messages([("system", map_template)])
map_chain = map_prompt | LLM | StrOutputParser()

if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """

    result = map_chain.invoke({"context": test_document})

    print("Generated Summary:")
    print(result)
