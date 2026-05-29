# 🩺 AI Prescription Generator

> **RAG-powered clinical decision support system** that retrieves relevant medical knowledge and generates structured prescriptions using LLMs — bridging the gap between patient data and evidence-based treatment recommendations.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Project Outcome

The AI Prescription Generator is a working prototype that demonstrates how **Retrieval-Augmented Generation (RAG)** can assist clinicians by:

- **Ingesting** patient data from PDF reports or manual forms (symptoms, history, allergies, demographics).
- **Retrieving** the top-3 most semantically relevant medical knowledge chunks from a pre-indexed FAISS vector store using HuggingFace sentence embeddings.
- **Generating** a structured prescription (diagnosis, justification, medication, dosage, duration, precautions, follow-up) via **Groq's Llama-4 Maverick** (17B parameters, 128K context).
- **Delivering** the output through an intuitive Streamlit interface with downloadable TXT reports.

### Sample Output
| Field | Generated Content |
|---|---|
| Diagnosis | Identified based on patient symptoms + retrieved literature |
| Reasoning | Justification grounded in the retrieved medical chunks |
| Prescription | Medication name, dosage, duration |
| Precautions | Drug interactions, allergy warnings |
| Follow-up | Recommended tests & revisit timeline |

---

## 📊 Impact & Metrics

| Metric | Value / Estimate |
|---|---|
| **Medical Knowledge Indexed** | PDF documents covering common conditions, drugs & guidelines |
| **Vector Search Speed** | FAISS L2 similarity — sub-millisecond retrieval for top-k queries |
| **Embedding Model** | `all-MiniLM-L6-v2` — 384-dimensional dense vectors, 80+ MB compressed |
| **LLM Inference** | Groq LPU™ infrastructure — ~300 tokens/sec average generation |
| **LLM Context Window** | 128,000 tokens — can handle lengthy patient histories + large retrieved contexts |
| **Time Saved per Consultation** | ~2–5 minutes (automated knowledge retrieval + draft prescription) |
| **Hallucination Mitigation** | Explicit prompt engineering ("Do not hallucinate. If uncertain, say so.") |
| **Accessibility** | Web-based (Streamlit) — runs on any device with a browser |

### Clinical Impact Potential
- ⚡ **Reduces cognitive load** — clinicians don't need to manually recall drug interactions or guidelines.
- 🔍 **Evidence-grounded** — every prescription is anchored to retrieved medical literature, not just LLM parametric memory.
- 🌐 **Scalable** — can be extended to multiple languages, specialties, or integrated into EHR systems.
- 📝 **Auditable** — retrieved context and generated output are both visible, enabling peer review.

---

## 🧱 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive web UI |
| **LLM** | [Groq](https://groq.com) + Llama-4 Maverick 17B | Prescription generation |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` via `langchain-huggingface` | Semantic search vectors |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatL2) | Similarity search |
| **Metadata** | Python `pickle` | Chunk storage |
| **PDF Parsing** | `PyPDF2` | Patient report ingestion |
| **Environment** | `python-dotenv` | API key & secret management |
| **Chunking** | Custom fixed-size (500 chars) | Document segmentation |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app3.py)                         │
│  ┌──────────────────┐    ┌──────────────────────────────────────────┐ │
│  │  Patient Data In  │    │  Generated Prescription Out             │ │
│  │  • PDF Upload     │    │  • Diagnosis + Reasoning               │ │
│  │  • Manual Form    │    │  • Medication, Dosage, Duration         │ │
│  └────────┬──────────┘    │  • Precautions + Follow-up             │ │
│           │               │  • Download as TXT                     │ │
│           ▼               └────────────────▲───────────────────────┘ │
│  ┌──────────────────┐                       │                         │
│  │ HuggingFace       │                       │                         │
│  │ Embedding Model   │                       │                         │
│  └────────┬──────────┘                       │                         │
│           │ query vector                     │                         │
│           ▼                                  │                         │
│  ┌──────────────────┐                       │                         │
│  │ FAISS Index       │──── top-3 chunks ──► │                         │
│  │ (L2 similarity)   │                       │                         │
│  └───────────────────┘                       │                         │
│                                              │                         │
│  ┌───────────────────────────────────────────┴──────────┐              │
│  │              GROQ API (Llama-4 Maverick 17B)          │              │
│  │     Prompt = Patient Data + Retrieved Context         │              │
│  └──────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│   OFFLINE INDEXING (generatingVectorEmbedding) │
│  Medical PDFs → Chunking (500 chars) →       │
│  HuggingFace Embeddings → FAISS Index        │
│  → chunk_metadata3.pkl + medical_index3.faiss│
└─────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com) (free tier available)
- [HuggingFace Token](https://huggingface.co/settings/tokens) (free)

### 1. Clone the Repository
```bash
git clone https://github.com/Aditya365x/AI_PRESCRIPTION_GENERATOR.git
cd AI_PRESCRIPTION_GENERATOR
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_groq_api_key_here
HF_TOKEN=hf_your_huggingface_token_here
```

### 4. Build the Medical Knowledge Index (First Time Only)
```bash
python generatingVectorEmbeddings.py
```
> This reads medical PDFs from the configured directory, chunks them, creates embeddings, and saves `medical_index3.faiss` + `chunk_metadata3.pkl`.

### 5. Launch the App
```bash
streamlit run app3.py
```
Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
AI_PRESCRIPTION_GENERATOR/
├── app3.py                         # Main Streamlit application
├── generatingVectorEmbeddings.py   # Offline indexing script
├── requirements.txt                # Python dependencies
├── medical_index3.faiss            # FAISS vector index
├── chunk_metadata3.pkl             # Chunk text metadata
├── .env                            # API keys (gitignored)
├── .devcontainer/                  # VS Code dev container config
└── README.md                       # This file
```

---

## 🔮 Future Scope & Improvements

| Feature | Description |
|---|---|
| **Multilingual Support** | Extend to Hindi, regional Indian languages |
| **Fine-tuned Medical Embeddings** | Replace `all-MiniLM-L6-v2` with PubMedBERT / BioClinicalBERT |
| **EHR Integration** | FHIR/HL7 compatible API for hospital systems |
| **Structured JSON Output** | Parse LLM output into machine-readable JSON for EHR ingestion |
| **Drug Interaction Checker** | Cross-reference prescriptions against drug databases |
| **Multi-turn Dialogue** | Allow clinicians to refine the prescription iteratively |
| **Authentication & Audit Logs** | HIPAA-compliant access control & logging |
| **PDF Prescription Export** | Generate formatted PDFs instead of TXT |
| **Containerization** | Docker + docker-compose for one-command deployment |

---

## ⚠️ Disclaimer

This tool is a **research prototype** and is **NOT intended for clinical use** without proper validation. Always consult a licensed medical professional before making any healthcare decisions.

---

## 👨‍💻 Author

**Aditya** — [GitHub](https://github.com/Aditya365x)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
