# AI-Powered FAQ Assistant

This project implements an AI-powered FAQ assistant using Python, FAISS for retrieval, and a Generative AI model for response generation.

## Features
- Load & clean FAQs (columns: prompt, response)
- Embedding retriever (Sentence-Transformers) + FAISS index
- Optional OpenAI RAG answer constrained to retrieved context
- Baseline TF-IDF retriever (for comparison)
- Evaluation: self-retrieval on prompts

## Repository Structure
- `app.py` → API / Streamlit app
- `faqs.csv` → FAQ dataset
- `notebooks/FAQ.ipynb` → Development notebook
- `docs/assignment.md` → Assignment description
- `requirements.txt` → Dependencies

## Setup
```bash
git clone <your-repo-link>
cd ai-faq-assistant
pip install -r requirements.txt
