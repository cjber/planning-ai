import polars as pl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import GPT4o

with open("./planning_ai/eval/eval.txt", "r") as f:
    compare_template = f.read()

with open("./planning_ai/eval/summary.txt", "r") as f:
    summary_template = f.read()


class SummaryEvaluator(BaseModel):
    score: int = Field(..., description="The number of the best summary.")


SLLM = GPT4o.with_structured_output(SummaryEvaluator, strict=True)

compare_prompt = ChatPromptTemplate([("system", compare_template)])
compare_chain = compare_prompt | SLLM

summary_prompt = ChatPromptTemplate([("system", summary_template)])
summary_chain = summary_prompt | GPT4o | StrOutputParser()


original = pl.read_parquet(Paths.STAGING / "gcpt3.parquet").filter(
    pl.col("attachments_id").is_null()
)
summaries1 = original[["text", "representations_summary"]].unique().head(20)

summaries2 = summaries1[["text"]]
summaries2 = summaries2.with_columns(
    pl.col("text")
    .map_elements(
        lambda x: summary_chain.invoke({"content": x}), return_dtype=pl.String
    )
    .alias("summary")
)

summaries = summaries1.join(summaries2, on="text")
summaries = summaries.with_columns(
    pl.struct(["text", "representations_summary", "summary"])
    .map_elements(
        lambda x: compare_chain.invoke(
            {
                "document": x["text"],
                "summary_1": x["representations_summary"],
                "summary_2": x["summary"],
            }
        ).score,
        return_dtype=pl.Int8,
    )
    .alias("score")
)
summaries["score"].value_counts()
