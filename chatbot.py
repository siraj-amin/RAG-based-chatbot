# chatbot.py - Streamlit-ready RAG chatbot with persistent vectorstore

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# -----------------------------
# HUGGINGFACE API TOKEN
# -----------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# -----------------------------
# STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="RAG Chatbot")
st.title("🤖 My RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# LOAD PDF AND VECTORSTORE
# -----------------------------
PDF_FILE = "sdg.pdf"
VECTORSTORE_DIR = "./chroma_db"

# Only create embeddings if vectorstore does not exist
if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
    vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
else:
    # Load PDF
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    split_docs = []
    for doc in docs:
        split_docs.extend(text_splitter.split_text(doc.page_content))

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )

# Retriever
retriever = vectorstore.as_retriever()

# -----------------------------
# PROMPT AND LLM SETUP
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
""")

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.3,
    max_new_tokens=512
)
llm = ChatHuggingFace(llm=endpoint)

# -----------------------------
# HANDLE USER INPUT
# -----------------------------
if question := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": question})

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    # Create prompt + call LLM
    answer = llm.invoke(prompt.format({"context": context, "question": question}))

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
