# chatbot.py - Streamlit-ready RAG chatbot with enhanced UI/UX

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# -----------------------------
# CONFIG
# -----------------------------
PDF_FILE = "sdg.pdf"
VECTORSTORE_DIR = "./chroma_db"

# HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# -----------------------------
# STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="RAG Chatbot 🤖", layout="wide", page_icon="🤖")

st.markdown("""
<div style="text-align: center;">
    <h1 style="color:#4B9CD3;">🤖 My RAG Chatbot</h1>
    <p style="color:#6C757D; font-size:16px;">Ask any question about your documents and get concise answers instantly!</p>
</div>
""", unsafe_allow_html=True)

chat_container = st.container()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# LOAD OR BUILD VECTORSTORE
# -----------------------------
try:
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
    else:
        loader = PyPDFLoader(PDF_FILE)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        split_docs = []
        for doc in docs:
            split_docs.extend(text_splitter.split_text(doc.page_content))

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
except Exception as e:
    st.error(f"Error loading PDF or vectorstore: {e}")
    st.stop()

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
# FUNCTION TO GET ANSWER
# -----------------------------
def get_answer(question):
    """Retrieve context and get answer from LLM."""
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        else:
            docs = retriever.retrieve(question)

        context = "\n\n".join([d.page_content for d in docs])
        answer = llm.invoke(prompt.format({"context": context, "question": question}))
    except Exception as e:
        answer = f"Error retrieving answer: {e}"
    return answer

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Type your question here...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show spinner while LLM processes
    with st.spinner("Generating answer... 🤖"):
        answer = get_answer(user_input)

    # Display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# DISPLAY CHAT HISTORY WITH AVATARS
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with chat_container:
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:10px 0;">
                <div style="background-color:#D1E7DD; padding:10px 15px; border-radius:15px; max-width:70%;">
                    <strong>You:</strong> {msg["content"]}
                </div>
                <div style="margin-left:10px;">👤</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with chat_container:
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:10px 0; justify-content:flex-end;">
                <div style="margin-right:10px;">🤖</div>
                <div style="background-color:#E2E3E5; padding:10px 15px; border-radius:15px; max-width:70%;">
                    <strong>Bot:</strong> {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
