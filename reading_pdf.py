import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    return splitter.split_text(text)

def get_vector_store_pdf(chunks,supabase,api_key,name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = SupabaseVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        client=supabase,
        table_name="document_chunks",
        metadatas=[{"source": name}] * len(chunks)
    )
    print(vector_store)
    print("Supabase Vector DB successfully updated!")