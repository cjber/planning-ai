# Planning AI

![Python](https://img.shields.io/badge/LangChain-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![LangChain](https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?style=for-the-badge&logo=openai&logoColor=white)

## Project Overview

Planning AI is a sophisticated tool designed to process and analyze responses to local government planning applications. It leverages advanced natural language processing capabilities to summarize and categorize feedback, providing insights into public opinion on proposed developments.

## Features

- **Document Processing**: Extracts and processes text from various document formats including PDFs and Excel files.
- **Summarization**: Generates concise summaries of responses, highlighting key points and overall sentiment.
- **Thematic Analysis**: Breaks down responses into thematic categories, providing a percentage breakdown of themes.
- **Rate Limiting**: Ensures API requests are managed efficiently to comply with usage limits.

## Installation

To set up the project, ensure you have Python 3.8 or higher installed. Then, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd planning_ai
pip install -r requirements.txt
```

## Usage

1. **Preprocessing**: Run the preprocessing scripts to convert raw data into a format suitable for analysis.
   ```bash
   python planning_ai/preprocessing/process_pdfs.py
   python planning_ai/preprocessing/gclp.py
   python planning_ai/preprocessing/web_comments.py
   ```

2. **Run Analysis**: Execute the main script to process the documents and generate summaries.
   ```bash
   python planning_ai/main.py
   ```

## Workflow

1. **Data Loading**: Documents are loaded from the staging directory using the `DirectoryLoader`.
2. **Text Splitting**: Documents are split into manageable chunks using `CharacterTextSplitter`.
3. **Graph Processing**: The `StateGraph` orchestrates the flow of data through various nodes, including mapping and reducing summaries.
4. **Summarization**: The `map_chain` and `reduce_chain` are used to generate and refine summaries.
5. **Output**: Final summaries and thematic breakdowns are printed and can be exported for further analysis.

## Configuration

- **Environment Variables**: Use a `.env` file to store sensitive information like API keys.
- **Constants**: Adjust `Consts` in `planning_ai/common/utils.py` to modify token limits and other settings.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact [Your Name] at [Your Email].
