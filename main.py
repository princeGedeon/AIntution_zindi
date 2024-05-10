# -*- coding: utf-8 -*-
"""main.py - Main script for loading vector database and performing inference."""

import os
import glob
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from ingestion import ingest_documents  # Assuming ingestion.py is properly set up to be imported

load_dotenv()

def ingest_documents_from_folder(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    for pdf_file in pdf_files:
        ingest_documents(pdf_file)

def main(query):
    # Ingest documents from the 'data' folder
    data_folder = './data'
    if not os.path.exists("vectorstore.db"):  # Check if the database does not exist
        ingest_documents_from_folder(data_folder)

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
    main("un petit résumé")
