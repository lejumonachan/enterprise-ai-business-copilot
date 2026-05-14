# 🤖 Enterprise AI Business Copilot

Enterprise AI Business Copilot is a premium AI-powered analytics platform designed for intelligent business data analysis, document understanding, AutoML workflows, dashboard generation, and executive reporting.

The platform supports CSV, Excel, and PDF uploads, integrates local RAG-based document intelligence using FAISS and Sentence Transformers, and provides interactive AI-driven insights through a professional Streamlit interface.

---

# 🚀 Features

## 📂 File Upload & Preview
- Upload CSV, Excel, and PDF files
- Automatic dataset profiling
- PDF text extraction

## 💬 AI Chat With Data
- Ask questions about uploaded datasets
- Chat with PDF documents using local RAG
- AI-powered contextual analysis

## 📊 Business Dashboard
- Interactive visual analytics
- Bar charts
- Line charts
- Pie charts
- Histograms
- Scatter plots
- Heatmaps

## 🤖 AutoML Engine
- Automatic ML workflow
- Model comparison
- Best model selection
- Live prediction console

## 🧠 Local RAG Document Intelligence
- PDF chunking
- Sentence-transformer embeddings
- FAISS vector search
- Context-aware retrieval

## 📄 Executive Report Generation
- AI-generated business insights
- Downloadable PDF reports

---

# 🛠️ Tech Stack

## Frontend
- Streamlit

## Data Processing
- Pandas
- NumPy

## Visualization
- Plotly

## Machine Learning
- Scikit-learn

## Local AI & RAG
- HuggingFace Transformers
- Sentence Transformers
- FAISS

## Document Processing
- PyPDF2

---

# 📁 Project Structure

```bash
enterprise_ai_business_copilot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── styles.css
│
├── utils/
│   ├── ai_engine.py
│   ├── automl_engine.py
│   ├── file_loader.py
│   ├── helpers.py
│   ├── pdf_generator.py
│   ├── rag_engine.py
│   └── visualization.py
│
├── uploads/
├── reports/
└── vector_store/
