import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import LangChainLLM

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define Groq's Llama 3 model using LangChain's ChatGroq
class GroqLlamaLLM(LangChainLLM):
    def __init__(self, model="llama3-8b-8192", temperature=0.1):
        self.model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model,
            temperature=temperature
        )

# Function to fetch and clean text from a URL
def fetch_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text_content = "\n".join([p.get_text() for p in paragraphs])
        return text_content[:5000]  # Limit to 5000 characters for processing
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"

# Streamlit UI
st.title("Web Page Q&A with Groq’s Llama 3")

# User input for URL
url = st.text_input("Enter a URL to extract content:")
if url:
    with st.spinner("Fetching content..."):
        page_content = fetch_url_content(url)
    
    if page_content.startswith("Error"):
        st.error(page_content)
    else:
        st.success("Content extracted successfully!")

        # Create a document and vector index
        document = Document(text=page_content)
        index = VectorStoreIndex.from_documents([document])
        index.storage_context.persist(persist_dir="./vectorstore")

        # Set up chat engine
        llm = GroqLlamaLLM()
        service_context = ServiceContext.from_defaults(llm=llm)
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

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
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
