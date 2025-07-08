import os
import pickle
import faiss
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ['HF_TOKEN'] = HF_TOKEN

# Load PDFs
def load_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            docs.append(text)
    return docs

# Chunk text into fixed-size segments
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Load and chunk documents
docs = load_pdfs("D:\LangchainProjects\medicinesDATA")
chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc))

# Generate embeddings via Hugging Face API
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]

# Convert to numpy array
import numpy as np
embeddings = np.array(embeddings)

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "medical_index3.faiss")

# Save metadata (chunk text)
with open("chunk_metadata3.pkl", "wb") as f:
    pickle.dump(chunks, f)
