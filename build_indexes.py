import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    JSONLoader
)

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

CATEGORY_FOLDERS = {
    "hr": "data/hr",
    "offline_reports": "data/offline_reports",
    "online_reports": "data/online_reports"
}

def load_all_docs_from_folder(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        ext = file.lower()

        try:
            if ext.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
            elif ext.endswith(".docx") or ext.endswith(".doc"):
                docs.extend(UnstructuredWordDocumentLoader(path).load())
            elif ext.endswith(".xlsx"):
                docs.extend(UnstructuredExcelLoader(path).load())
            elif ext.endswith(".json"):
                docs.extend(JSONLoader(file_path=path, jq_schema=".[]", text_content=False).load())
        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")
    return docs
def build_faiss_index(category, folder_path):
    print(f"📂 Building index for {category} from {folder_path}")
    docs = load_all_docs_from_folder(folder_path)
    if not docs:
        print(f"⚠️ No documents found in {folder_path}")
        return

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    index_path = f"faiss_index_{category}"
    faiss_file = os.path.join(index_path, "index.faiss")

    if os.path.exists(faiss_file):
        print(f"📌 Updating existing FAISS index for {category}")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        print(f"🆕 Creating new FAISS index for {category}")
        db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)
    print(f"✅ Index saved to {index_path}")

if __name__ == "__main__":
    for cat, folder in CATEGORY_FOLDERS.items():
        build_faiss_index(cat, folder)
