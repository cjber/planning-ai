from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader("./data/raw/pdfs/57693-94 Response Form.pdf")
loader.load()
