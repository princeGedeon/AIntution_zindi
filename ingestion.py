from tools.utils import  ingest_documents_pdf_from_folder, ingest_documents_docx_from_folder

if __name__ == "__main__":
    data_folder="./data"
    ingest_documents_pdf_from_folder(data_folder)
    ingest_documents_docx_from_folder(data_folder)
