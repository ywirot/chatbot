
# rag_streamlit_chatbot.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# === CONFIG ===
os.environ["Groq-img"] = "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"  # 🔑 เปลี่ยนตรงนี้

st.set_page_config(page_title="📄 RAG Chatbot from PDF", page_icon="🤖")

st.title("📄🤖 RAG Chatbot from PDF (Groq + LLaMA3)")
st.markdown("ถามคำถามจากไฟล์ PDF โดยใช้ RAG และ LLaMA3 จาก Groq")

uploaded_file = st.file_uploader("📤 อัปโหลด PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("📚 กำลังประมวลผลเอกสาร..."):
        # === Step 1: Load PDF ===
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp_uploaded.pdf")
        pages = loader.load()

        # === Step 2: Split Document ===
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # === Step 3: Embed & Store ===
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # === Step 4: Set up Groq + RetrievalQA ===
        llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        st.success("✅ เอกสารถูกโหลดและฝังเรียบร้อยแล้ว!")

        # === Step 5: Chat Interface ===
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.text_area("✍️ ถามคำถามที่เกี่ยวข้องกับเอกสาร")

        if st.button("ถาม"):
            if user_question.strip() != "":
                with st.spinner("🤖 กำลังค้นและตอบ..."):
                    result = rag_chain.invoke({"query": user_question})
                    answer = result["result"]

                    st.session_state.chat_history.append(("คุณ", user_question))
                    st.session_state.chat_history.append(("Bot", answer))

        # Show chat history
        for sender, message in reversed(st.session_state.chat_history):
            st.markdown(f"**{sender}:** {message}")
else:
    st.info("📌 กรุณาอัปโหลดไฟล์ PDF เพื่อเริ่มต้น")
