
# Zindi AI RAG Documentation

## Introduction
This project involves two main scripts for managing a vector database with document embeddings and querying them using language models:
- `ingestion.py`: Script for ingesting documents into a vector database.
- `main.py`: Main script for loading the vector database and performing inference.

## Setup Instructions

### Requirements
Ensure you have Python installed on your machine. The project is tested with Python 3.7 and above. You will need the following packages:
- `langchain`
- `langchain_community`
- `langchain_core`
- `faiss-cpu` for FAISS (use `faiss-gpu` if GPU is available)

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
