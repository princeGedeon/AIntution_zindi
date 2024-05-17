
# Zindi AI RAG Documentation

## Introduction
This project involves two main scripts for managing a vector database with document embeddings and querying them using language models

## Directory Structure

```plaintext
.
├── data
│   ├── document1.pdf
│   ├── document2.docx
│   └── ...
├── test_files
│   ├── contexts.csv
│   └── ...
├── tools
│   ├── utils.py
│   └── ...
├── main.py
├── zindi.py
├── requirements.txt
└── README.md
```

### Requirements
Ensure you have Python installed on your machine. The project is tested with Python 3.7 and above. You will need the following packages:
- `langchain`
- `langchain_community`
- `langchain_core`
- `faiss-cpu` for FAISS (use `faiss-gpu` if GPU is available)

### Prerequisites 
- Create a .env file in the root directory of the project and provide your API keys:
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install required Python packages:
   ```bash
    streamlit run app.py
   ```




## Instructions

### Step 1: Ingest Documents

Place all the documents (PDF and DOCX formats) that you want to ingest into the vector database inside the `data` folder. The ingestion script will process these documents and populate the vector database.

### Step 2: Execute the Ingestion Script

Run the following command to ingest documents and create the vector database:

```bash
python ingestion.py
```

This script will:
- Check if the vector database already exists. If not, it will ingest documents from the `data` folder.
- Initialize the embedding model using HuggingFace embeddings.
- Load the vector database using FAISS.

### Step 3: Retrieve Relevant Contexts

After the ingestion process, you can retrieve relevant contexts by executing the `zindi.py` script. This script will generate relevant contexts based on your query and store them in a CSV file inside the `test_files` folder.

Run the following command to execute the context retrieval:

```bash
python zindi.py
```
This script:
- Takes a query as input.
- Retrieves the top 5 relevant contexts from the vector database.
- Stores these contexts in a CSV file inside the `test_files` folder.


