# **HR Chatbot with RAG**

An intelligent HR chatbot that conducts initial screening interviews using Retrieval-Augmented Generation (RAG) to personalize conversations based on candidate resumes stored in a vector database.

## **Prerequisites**
- Python 3.9+
- Pinecone API key
- Groq API key
- `.env` file with your credentials

**Sample `.env` file:**
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
PINECONE_ENVIRONMENT=your_pinecone_environment
GROQ_MODEL_NAME=your_groq_model_name
```

## **Installation**
1. **Clone the repository:**
  ```bash
  git clone https://github.com/anubhavlal07/RAG-Agent.git
  cd RAG-Agent
  ```
2. **Create and activate a virtual environment:**
  ```bash
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On macOS/Linux:
  source venv/bin/activate
  ```
3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
4. **Create a `.env` file with your credentials** (see sample above).

## **Running the Application**

### 1. Downloading and Running the Parser API

If your workflow requires parsing resumes or documents, start the parser API:

```bash
cd resumeparser
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

This will launch the parser API server. Make sure it is running before proceeding to data ingestion if parsing is needed.

### 2. Data Ingestion

To ingest candidate resumes into the vector database, run:

```bash
cd agent
python ingest.py
```

This script will process and upload the data to Pinecone.

### 3. Running the Interview Chatbot

To start the HR chatbot for conducting interviews:

```bash
python main.py
```

This will launch the chatbot interface, ready to interact with candidates.

---

**Note:** Ensure your `.env` file is correctly configured and all services (like the parser API) are running as needed before starting the main application.
