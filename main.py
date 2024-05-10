# -*- coding: utf-8 -*-
"""main.py - Main script for loading vector database and performing inference."""

import os
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def main(query):
    # Load the vectorstore
    vectorstore = FAISS.load_local("vectorstore.db")

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
