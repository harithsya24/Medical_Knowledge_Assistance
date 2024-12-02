# AgentRAG: Medical Knowledge Retrieval and Generation System

AgentRAG is an advanced medical knowledge retrieval and generation system that combines **retrieval-augmented generation (RAG)** for assisting medical practitioners. It uses the **FAISS** index for efficient document retrieval and a **GPT-2** model for generating responses based on retrieved documents. The system is designed to support medical practitioners with up-to-date clinical guidelines, disease management protocols, and relevant research findings.

## Features
- **Document Retrieval**: Uses FAISS to retrieve the most relevant documents from a large set of medical texts based on a user's query.
- **Response Generation**: Generates contextually relevant answers to queries using GPT-2, augmented with retrieved documents.
- **External Tool Integration**: Supports querying external databases (e.g., PubMed) for further information if necessary.
- **Medical Knowledge**: Leverages a large medical dataset and clinical guidelines to answer questions related to drug interactions, treatment protocols, symptoms, and more.
- **Streamlit Interface**: Interactive UI that allows medical practitioners to input queries and receive answers instantly.

## Requirements
1. Python 3.7 or higher
2. Streamlit
3. `transformers`
4. `sentence-transformers`
5. `faiss-cpu` or `faiss-gpu` (for indexing)
6. `datasets`
7. `requests` (for external database querying)
