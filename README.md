# 🧠 NeuroAssistant
NeuroAssistant is an AI-powered medical chatbot specialized in neurology and psychiatry. It uses RAG (Retrieval-Augmented Generation) to answer user questions by retrieving relevant context from a medical textbook and generating responses using a large language model.

## Project Objective
The goal is to assist users — especially medical students or patients—with reliable, context-based answers to neurological and psychiatric questions.

## Tech Stack
- **LangChain** – for document loading, text splitting, embeddings, and retrieval.
- **Transformers** – to load and run google/flan-t5-base for generating answers.
- **FAISS** – for vector similarity search.
- **Gradio** – for creating an interactive chatbot interface.
- **PyPDFLoader** – to extract text from the PDF medical textbook.
- **HuggingFace Sentence Transformers** – for embeddings (`all-MiniLM-L6-v2`).


## How It Works

1. The medical textbook (`Diseases_of_The_Brain.pdf`) is loaded and split into chunks.
2. Each chunk is converted into embeddings using all-MiniLM-L6-v2.
3. Chunks are stored in a FAISS vector database.
4. When the user asks a question:
   - The retriever fetches the most relevant chunk(s).
   - A prompt is constructed combining the user's question and the retrieved context.
   - The FLAN-T5 model generates the answer based on this prompt.

