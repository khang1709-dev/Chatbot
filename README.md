# Financial RAG Chatbot

> *An AI-powered assistant for querying financial reports using Retrieval-Augmented Generation (RAG).*

## Overview
This project builds an intelligent chatbot capable of reading, understanding, and answering questions based on user-provided financial documents (PDFs, Excel, etc.). 
By leveraging **RAG (Retrieval-Augmented Generation)**, the system solves the common "hallucination" problem of Large Language Models (LLMs) by grounding every answer in actual data.

## Key Features
* **Smart Chunking:** Automatically processes and chunks raw text from financial reports for optimal retrieval.
* **Semantic Search:** Utilizes a Vector Database to retrieve the most relevant context based on user queries.
* **Source Citation:** Provides references (file name, page number) for every answer to ensure transparency.

## Tech Stack
* **Language:** Python 3.10+
* **Embedding Model:** HuggingFace (BGE-M3 / All-MiniLM)
* **Vector Database:** ChromaDB
* **LLM Integration:** Groq API / HuggingFace Hub
* **Libraries:** LangChain, python-dotenv, Pandas

## Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/khang1709-dev/Chatbot.git](https://github.com/khang1709-dev/Chatbot.git)
cd Chatbot
