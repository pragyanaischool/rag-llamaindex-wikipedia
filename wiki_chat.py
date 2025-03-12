import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.response.pprint_utils import pprint_response
from llama_index.llms import Groq
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.readers import WikipediaReader

# Load environment variables
load_dotenv()

# Set up vector store path
storage_path = "./vectorstore"

# Use Groq LLM instead of OpenAI
llm = Groq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
service_context = ServiceContext.from_defaults(llm=llm)

# Load Wikipedia data
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Star Wars Movie', 'Star Trek Movie'])

# Create and persist vector store index
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=storage_path)

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
            pprint_response(response, show_source=True)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
