import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ----------------------------------------------

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

CATEGORY_INDEXES = {
    "HR": "faiss_index_hr",
    "Offline Reports": "faiss_index_offline_reports",
    "Online Reports": "faiss_index_online_reports"
}

# Updated prompt to ensure all results are returned
prompt_template = """
You are a helpful assistant. Based on the context from the retrieved documents,
List every matching report name without omitting any.
If multiple matches exist, return them as a bullet list.
Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def query_category(category, query):
    index_path = CATEGORY_INDEXES[category]
    if not os.path.exists(index_path):
        return f"❌ No index found for {category}. Please build it first."

    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 35})  # Increased recall

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain.run(query)

# ---- Streamlit UI ----
st.set_page_config(page_title="Category-Specific RAG", page_icon="🤖", layout="centered")
st.title("📂 Category-Specific RAG Chatbot")

category = st.selectbox("Select Category", list(CATEGORY_INDEXES.keys()))
query = st.text_input("Ask your question:")

if query:
    with st.spinner(f"Searching in {category} knowledge base..."):
        answer = query_category(category, query)
        st.write("**Answer:**", answer)
