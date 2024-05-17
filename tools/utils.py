import glob
import os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents_pdf_from_folder(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    for pdf_file in pdf_files:
        ingest_pdf(pdf_file)


def ingest_documents_docx_from_folder(folder_path):
    docs = glob.glob(os.path.join(folder_path, '*.docx'))
    for doc in docs:
        ingest_docx(doc)


def ingest_docx(file_path, model_name='BAAI/bge-small-en-v1.5'):
    # Load a PDF
    loader = Docx2txtLoader(file_path)
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
    print("Documents words ingested and vector database saved.")

def ingest_pdf(file_path, model_name='BAAI/bge-small-en-v1.5'):
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
    print("Documents pdf ingested and vector database saved.")