import streamlit as st
import os
import requests
from dotenv import load_dotenv
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.llms.base import LLM
from llama_index.llms.custom import CustomLLM
from llama_index.readers import WikipediaReader

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom Groq API Wrapper
class GroqLLM(LLM):
    def __init__(self, model="llama3-8b-8192", temperature=0.1):
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        url = "https://api.groq.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "system", "content": "You are an AI assistant."},
                         {"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        response = requests.post(url, json=data, headers=headers)
        return response.json()["choices"][0]["message"]["content"]

# Use the custom Groq LLM in ServiceContext
llm = GroqLLM()
service_context = ServiceContext.from_defaults(llm=llm)

# Load Wikipedia data
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Star Wars Movie', 'Star Trek Movie'])

# Create and persist vector store index
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./vectorstore")

# Streamlit UI
st.title("Ask the Wiki On Star Wars")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

# Create chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Chat interface
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
