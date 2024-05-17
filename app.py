import streamlit as st
import os

from dotenv import load_dotenv

from tools.main import main
from tools.utils import ingest_pdf

load_dotenv() 
# Set the title of the app
st.title('Document Ingestion and Query Interface')

# Sidebar for document upload and ingestion
st.sidebar.title("Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type='pdf')
if uploaded_file is not None:
    file_path = f"./temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if st.sidebar.button('Ingest Document'):
        ingest_pdf(file_path)
        st.sidebar.success('Document ingested successfully.')
        os.remove(file_path)  # Clean up the uploaded file

# Main panel for querying
st.header("Query the Document Database")
user_query = st.text_input("Enter your query here", "")
if st.button('Get Answer'):
    if user_query:
        response = main(user_query)
        st.write("Response:", response)
    else:
        st.write("Please enter a query to get a response.")
