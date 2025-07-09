import streamlit as st
import faiss
import pickle
import numpy as np
import requests
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
dotenv_path = ".env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# Load FAISS index and metadata
index = faiss.read_index("D:/LangchainProjects/PrescriptionGEN_PROTOTYPE2/medical_index3.faiss")
with open("D:/LangchainProjects/PrescriptionGEN_PROTOTYPE2/chunk_metadata3.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model via Hugging Face inference API
model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Prescription Generator", layout="wide")
st.title("🩺 AI Prescription Generator")

st.markdown("Upload a patient file (PDF) or enter the data manually.")

# Initialize session state
if "process_pdf" not in st.session_state:
    st.session_state.process_pdf = False
if "patient_text" not in st.session_state:
    st.session_state.patient_text = ""

# Function to clean unwanted assistant artifacts

def clean_text(text):
    trigger = "COMMAND ---> generate the file pdf"
    if trigger in text:
        text = text.split(trigger)[0]
    return text.strip()

# Option 1: Upload PDF
uploaded_file = st.file_uploader("Upload Patient PDF", type="pdf")

if uploaded_file:
    if st.button("📄 Process Uploaded PDF"):
        reader = PdfReader(uploaded_file)
        patient_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        patient_text = clean_text(patient_text)
        st.session_state.patient_text = patient_text
        st.session_state.process_pdf = True

# Option 2: Manual Entry
with st.expander("Or enter patient details manually"):
    age = st.text_input("Age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    symptoms = st.text_area("Symptoms (comma separated)")
    medical_history = st.text_area("Medical History")
    allergies = st.text_area("Known Allergies")
    if st.button("Use Manual Data"):
        patient_text = f"Age: {age}\nGender: {gender}\nSymptoms: {symptoms}\nMedical History: {medical_history}\nAllergies: {allergies}"
        st.session_state.patient_text = patient_text
        st.session_state.process_pdf = True

# If patient text is available, generate prescription
if st.session_state.get("process_pdf") and st.session_state.get("patient_text"):
    patient_text = st.session_state.patient_text

    

    # Vector Search
    query_vector = model.embed_query(patient_text)
    D, I = index.search(np.array([query_vector]), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    with st.expander("📚 Retrieved Medical Context"):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Document {i+1}:**")
            st.code(chunk)

    # Create prompt
    prompt = f'''
            Patient data:
            {patient_text}

            Retrieved medical knowledge:
            {retrieved_chunks[0]}
            {retrieved_chunks[1]}
            {retrieved_chunks[2]}

            Based on the information above:
            1. Identify the most likely diagnosis.
            2. Justify your reasoning.
            3. Provide a prescription with medication name, dosage, duration, and necessary precautions.
            4. Mention follow-up recommendations or tests, if needed.

            Respond like a professional medical assistant. Do not hallucinate. If uncertain, say so.
        '''

    if st.button("🧠 Generate Prescription"):
        with st.spinner("Calling LLM to generate prescription..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                st.subheader("💊 Prescription")
                st.success("Prescription generated successfully.")
                st.text_area("📋 Full Prescription Output", value=result, height=300)

                # Download button
                st.download_button(
                    label="📥 Download Prescription as TXT",
                    data=result,
                    file_name="prescription.txt",
                    mime="text/plain"
                )
            else:
                st.error("Failed to get response from LLM.")

