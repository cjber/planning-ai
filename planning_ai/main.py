from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

from planning_ai.common.utils import Paths
from planning_ai.graph import create_graph


def main():
    loader = DirectoryLoader(
        path=str(Paths.STAGING),
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        recursive=True,
    )
    docs = [doc for doc in loader.load()[10:30] if doc.page_content]
    # TEMP: limit docs
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    app = create_graph()

    step = None
    for step in app.stream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))

    if step is None:
        raise ValueError("No steps were processed!")

    return step


if __name__ == "__main__":
    out = main()
    quarto_doc = (
        """---
title: "Semantic Data Catalogue"
format: 
  PrettyPDF-pdf:
        papersize: A4
execute:
  freeze: auto
  echo: false
monofont: 'JetBrains Mono'
monofontoptions: 
  - Scale=0.55
---\n\n"""
        + out["generate_final_summary"]["final_summary"]
    )

    with open("./reports/DEMO_REPORT.qmd", "w") as f:
        f.write(quarto_doc)
