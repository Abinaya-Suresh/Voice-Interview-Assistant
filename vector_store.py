from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_stores(python_file_path: str, sql_file_path: str):
    """
    Create separate FAISS vectorstores for Python and SQL books
    """
   
    python_loader = PyPDFLoader(r"C:\Users\abina\Desktop\Resume Tracker\Interview_prep\python_book.pdf")
    python_docs = python_loader.load()

    python_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000
    )
    python_texts = python_text_splitter.split_documents(python_docs)

    
    sql_loader = PyPDFLoader(r"C:\Users\abina\Desktop\Resume Tracker\Interview_prep\sql_book.pdf")
    sql_docs = sql_loader.load()

    sql_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000
    )
    sql_texts = sql_text_splitter.split_documents(sql_docs)

    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    
    python_vectorstore = FAISS.from_documents(python_texts, embeddings)
    sql_vectorstore = FAISS.from_documents(sql_texts, embeddings)

    return python_vectorstore, sql_vectorstore
