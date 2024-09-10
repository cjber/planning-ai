from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from planning_ai.chains.map_chain import BriefSummary
from planning_ai.llms.llm import LLM

with open("./planning_ai/chains/prompts/reduce.txt", "r") as f:
    reduce_template = f.read()

reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
reduce_chain = reduce_prompt | LLM | StrOutputParser()

if __name__ == "__main__":
    test_summary = """
        The author expresses concern over the proposed mass development north-west of Cambridge,
        highlighting significant growth in the area over the past twenty years, particularly with
        the creation of Cambourne and the expansion of Papworth Everard.
        Related Aims: [Homes, Infrastructure]
        """

    result = reduce_chain.invoke({"context": test_summary})

    print("Generated Report:")
    print(result)
