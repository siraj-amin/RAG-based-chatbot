import os
import time
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

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# -----------------------------
# STREAMLIT SETUP
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 My RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# LOAD VECTORSTORE
# -----------------------------
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

# -----------------------------
# PROMPT & LLM SETUP
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

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
# DISPLAY CHAT HISTORY WITH BUBBLES
# -----------------------------
def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:10px 0;">
                <div style="background-color:#D1E7DD; padding:10px 15px; border-radius:15px; max-width:70%;">
                    <strong>You:</strong> {msg["content"]}
                </div>
                <div style="margin-left:10px;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:10px 0; justify-content:flex-end;">
                <div style="margin-right:10px;">🤖</div>
                <div style="background-color:#E2E3E5; padding:10px 15px; border-radius:15px; max-width:70%;">
                    <strong>Bot:</strong> {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

display_messages()

# -----------------------------
# GET BOT ANSWER WITH TYPING ANIMATION
# -----------------------------
question = st.chat_input("Ask your question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    display_messages()  # show updated chat

    bot_placeholder = st.empty()  # placeholder for typing animation

    with st.spinner("Bot is typing... 🤖"):
        # Retrieve relevant docs using similarity_search
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([d.page_content for d in docs])

        # Call your existing RAG chain
        chain = prompt | llm
        answer = chain.invoke({"context": context, "question": question})

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Typing animation: display one character at a time
    typed_answer = ""
    for char in answer:
        typed_answer += char
        bot_placeholder.markdown(f"""
        <div style="display:flex; align-items:center; margin:10px 0; justify-content:flex-end;">
            <div style="margin-right:10px;">🤖</div>
            <div style="background-color:#E2E3E5; padding:10px 15px; border-radius:15px; max-width:70%;">
                <strong>Bot:</strong> {typed_answer}
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.02)  # adjust typing speed

    display_messages()  # ensure full answer is displayed at the end
