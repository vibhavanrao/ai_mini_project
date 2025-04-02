import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import time

#function to get the text from the url provided by the user by scraping with the help of selenium
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

#function to clean the text scraped from the website url
def clean_data(soup):
    for script in soup(['script', 'style', 'footer', 'header', 'nav', 'aside']):
        script.decompose()
    return soup.get_text(separator=' ', strip=True)

#function for storing the data extracted from the link into the vector database 
def get_vector_store_url(chunks,url,supabase,api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = SupabaseVectorStore.from_texts(
        chunks,
        embedding=embeddings,
        client=supabase,
        table_name="document_chunks",
        metadatas=[{"source": url, "chunk_id": i} for i in range(len(chunks))]
    )
    print(vector_store)