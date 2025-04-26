# RAG Agent

This repository contains a Python-based implementation of a Retrieval-Augmented Generation (RAG) Agent. The RAG Agent combines retrieval-based techniques with generative AI to provide accurate and context-aware responses. This project is designed to be modular, scalable, and easy to use.

## Features

- **Retrieval-Augmented Generation**: Combines retrieval of relevant documents with generative AI for enhanced responses.
- **Customizable Pipelines**: Easily adapt the agent to your specific use case.
- **Scalable Architecture**: Designed to handle large datasets and complex queries.
- **Extensible**: Add new retrieval methods or generative models as needed.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- `pip` (Python package manager)
- A virtual environment tool (optional but recommended)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/rag-agent.git
    cd rag-agent
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the RAG Agent using Streamlit:
    ```bash
    streamlit run app.py
    ```

## Environment Variables

Create a `.env` file in the root directory of the project and add the following content:

    ```env
    PINECONE_API_KEY='Api-Key'
    PINECONE_INDEX_NAME='rag-agent'
    PINECONE_ENV='us-east-1'
    GROQ_API_KEY='Api-Key'
    GROQ_MODEL_NAME='llama3-8b-8192'
    ```
    Ensure that the `.env` file is not committed to version control by adding it to your `.gitignore` file.