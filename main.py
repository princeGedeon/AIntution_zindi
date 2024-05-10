# -*- coding: utf-8 -*-
"""main.py - Main script for loading vector database and performing inference."""

import os
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv() 

def main(query):
     # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', encode_kwargs={"normalize_embeddings": True})

    # Load the vectorstore
    vectorstore = FAISS.load_local("vectorstore.db",embeddings, allow_dangerous_deserialization=True)

    # Create a retriever
    retriever = vectorstore.as_retriever()

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv('GOOGLE_API_KEY'))

    # Define prompt template
    template = """
    Vous êtes un assistant pour les tâches de réponse aux questions.
    Utilisez le contexte fourni uniquement pour répondre à la question suivante :

    <context>
    {context}
    </context>

    Question : {input}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create a retrieval chain with document combination
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    # Invoke the chain with user query
    response = chain.invoke({"input": query})
    print("Response:", response['answer'])

if __name__ == "__main__":
    main("un petit résumé")
