import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import time

def scrape_website(url):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3)
    page_source = driver.page_source
    driver.quit()
    soup = BeautifulSoup(page_source, 'html.parser')
    cleaned_text = clean_data(soup)
    return cleaned_text

def clean_data(soup):
    for script in soup(['script', 'style', 'footer', 'header', 'nav', 'aside']):
        script.decompose()
    return soup.get_text(separator=' ', strip=True)

def get_vector_store_url(chunks,url,supabase,api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = SupabaseVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        client=supabase,
        table_name="document_chunks",
        metadatas=[{"source": url}] * len(chunks)
    )
    print(vector_store)