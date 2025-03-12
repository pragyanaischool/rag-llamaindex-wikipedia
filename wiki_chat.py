import streamlit as st
import os
import requests
import asyncio
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import LangChainLLM
from llama_index.core.query_engine import RetrieverQueryEngine

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to fetch text from a URL
def fetch_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text_content = "\n".join([p.get_text() for p in paragraphs])
        return text_content[:5000]  # Limit to 5000 characters
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"

# Function to initialize Groq's Llama 3 with LangChain
def get_groq_llm(model="llama3-8b-8192", temperature=0.1):
    return LangChainLLM(
        llm=ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model,
            temperature=temperature
        )
    )

# Streamlit UI
st.title("RAG-Powered Web Q&A using Groq’s Llama 3 & Hugging Face")

# User inputs a URL
url = st.text_input("Enter a URL to extract content:")
if url:
    with st.spinner("Fetching content..."):
        page_content = fetch_url_content(url)

    if page_content.startswith("Error"):
        st.error(page_content)
    else:
        st.success("Content extracted successfully!")

        # Use Hugging Face embeddings
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create document and vector index
        document = Document(text=page_content)
        index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
        index.storage_context.persist(persist_dir="./vectorstore")

        # Initialize LLM & RAG Retriever
        llm = get_groq_llm()
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        retriever = index.as_retriever()
        query_engine = RetrieverQueryEngine(retriever=retriever, service_context=service_context)

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about the webpage!"}]

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant information..."):
                    response = query_engine.query(prompt)
                    st.write(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
