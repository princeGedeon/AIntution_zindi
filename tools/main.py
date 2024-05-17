# -*- coding: utf-8 -*-
"""main.py - Main script for loading vector database and performing inference."""

import os
import glob

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

from tools.utils import ingest_documents_pdf_from_folder, ingest_documents_docx_from_folder

load_dotenv()

def retrieve_contexts(query, num_contexts=6):
    # Ingest documents from the 'data' folder if the vectorstore doesn't exist
    data_folder = './data'
    if not os.path.exists("vectorstore.db"):
        ingest_documents_pdf_from_folder(data_folder)
        ingest_documents_docx_from_folder(data_folder)

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', encode_kwargs={"normalize_embeddings": True})

    # Load the vectorstore
    vectorstore = FAISS.load_local("vectorstore.db", embeddings, allow_dangerous_deserialization=True)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": num_contexts})

    compressor = CohereRerank(top_n=num_contexts)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    #
    compressed_docs = compression_retriever.get_relevant_documents(query)


    # Retrieve top N relevant documents
    #docs = retriever.get_relevant_documents(query)

    # Extract contexts from the retrieved documents
    contexts = [doc.page_content for doc in compressed_docs]

    return contexts

def main(query):
    # Ingest documents from the 'data' folder
    data_folder = './data'
    if not os.path.exists("vectorstore.db"):  # Check if the database does not exist
        ingest_documents_pdf_from_folder(data_folder)
        ingest_documents_docx_from_folder(data_folder)

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', encode_kwargs={"normalize_embeddings": True})

    # Load the vectorstore
    vectorstore = FAISS.load_local("vectorstore.db", embeddings, allow_dangerous_deserialization=True)

    # Create a retriever
    retriever = vectorstore.as_retriever()


    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv('GOOGLE_API_KEY'))

    # Define prompt template
    template = """
    Bonjour, je suis votre assistant virtuel spécialisé dans l'analyse de documents. Je suis ici pour fournir des informations précises basées sur les documents que j'ai analysés. 
    Voici les informations contextuelles pertinentes à votre question :

    <context>
    {context}
    </context>

    En tenant compte de ce contexte, comment puis-je vous aider avec la question suivante ?
    
    Question : {input}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create a retrieval chain with document combination
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    # Invoke the chain with user query
    response = chain.invoke({"input": query})
    print("Response:", response['answer']) 
    return response['answer']

if __name__ == "__main__":
    text=retrieve_contexts("what is law")
    print(text)
    print(len(text))
