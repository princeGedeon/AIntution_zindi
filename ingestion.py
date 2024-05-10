# -*- coding: utf-8 -*-
"""ingestion.py - Script for loading documents into a vector database."""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def ingest_documents(file_path, model_name='BAAI/bge-small-en-v1.5'):
    # Load a PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})

    # Create a vectorstore
    vectorstore = FAISS.from_documents(text, embeddings)

    # Save the documents and embeddings
    vectorstore.save_local("vectorstore.db")
    print("Documents ingested and vector database saved.")

if __name__ == "__main__":
    ingest_documents("./recueil.pdf")
