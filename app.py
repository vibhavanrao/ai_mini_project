import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from supabase import create_client
from reading_pdf import *
from reading_link import *
from streamlit_local_storage import LocalStorage

#loading the environment variables
load_dotenv()
api_key = os.getenv("GEN_AI_API_KEY")

#configuring the genai
genai.configure(api_key=api_key)

#giving the credentials for the supabase database
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

#initializing the local storage
ls = LocalStorage()

#delete_supabase_data function is created to delete the data in database so that it doesnot effect the output of the new searches
def delete_supabase_data():
    """
        here the contents of the data base are deleted where the id is not equal to 0
    """
    try:
        supabase.table("document_chunks").delete().neq("id", 0).execute()
        st.success("Supabase data deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting data from Supabase: {str(e)}")



def get_conversational_chain():
    prompt = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, respond with:
    'Answer is not available in the context.'

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    try:
        vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="document_chunks")
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        return {"output_text": f"Error fetching from Supabase: {str(e)}"}

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

st.set_page_config(page_title="PDF Chatbot")

# Initialize session state for URLs
if "urls" not in st.session_state or len(st.session_state.urls) == 0:
    stored_urls = ls.getItem('urls')
    stored_urls = stored_urls or []
    st.session_state.urls = stored_urls

if "messages" not in st.session_state:
    messages = LocalStorage().getItem('messages')
    stored_messages = messages or [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]
    st.session_state.messages = stored_messages
    

# Sidebar UI
with st.sidebar:
    st.title("📌 Chatbot Menu")

    # Upload PDFs
    with st.expander("Upload PDFs"):
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("📂 Process PDF"):
            with st.spinner("Processing..."):
                for doc in pdf_docs:
                    print(doc)
                    raw_text = get_pdf_text(doc)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store_pdf(text_chunks, supabase, api_key,doc.name)
                        st.success("Processing complete!")
                    else:
                        st.error("No text extracted from PDFs. Please try another file.")

    # Input for URL
    with st.expander("Scrape Website"):
        url_input = st.text_input("Enter a URL:")
        if st.button("🌐 Process URL"):
            if url_input.strip() and url_input not in st.session_state.urls:
                st.session_state.urls.append(url_input)  # Add URL to session state
                
                with st.spinner("Scraping the data from the URL..."):
                    raw_text = scrape_website(url_input)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store_url(text_chunks, url_input, supabase, api_key)
                        st.success("Processing complete!")
                        ls.setItem('urls', st.session_state.urls, key = 'set_url')
                        st.rerun()
                    else:
                        st.error("Error scraping the website.")
        for url in st.session_state.urls:
            st.write(f"- {url}")

    if st.button("🗑 Clear Database"):
        delete_supabase_data()
        st.session_state.urls = []
        ls.setItem('urls', st.session_state.urls, key='remove_url')
        st.rerun()
        
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}]
        ls.setItem('messages', st.session_state.messages)

# Initialize messages in session state

st.title("📄 AI Chatbot for PDFs & URLs")
st.write("Upload PDFs or enter a URL, and start asking questions!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    ls.setItem('messages', st.session_state.messages, key = 'set_questions')

# Generate assistant response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = user_input(prompt)
            placeholder = st.empty()
            full_response = response.get('output_text', 'Error generating response.')

            placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
            ls.setItem('messages', st.session_state.messages, key = 'set_messages')
