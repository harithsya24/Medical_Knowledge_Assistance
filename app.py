import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset

# Define the AgentRAG class
class AgentRAG:
    def __init__(self, faiss_index, text_data, embedder, model, tokenizer):
        self.faiss_index = faiss_index
        self.text_data = text_data
        self.embedder = embedder
        self.model = model
        self.tokenizer = tokenizer

    def retrieve_documents(self, query, k=3):
        query_embedding = self.embedder.encode([query]).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)
        return [self.text_data[i] for i in indices[0]]

    def generate_response(self, query, relevant_docs):
        relevant_docs = relevant_docs[:2]
        context = " ".join(relevant_docs) + " " + query
        inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=800,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize the models and FAISS index
@st.cache_resource
def initialize_agent():
    st.write("Loading dataset...")
    dataset = load_dataset("MedRAG/textbooks", split="train")
    text_data = [item['content'] for item in dataset]
    st.write("Dataset loaded!")

    st.write("Encoding dataset with SentenceTransformers...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(text_data)
    st.write(f"Encoded {len(embeddings)} embeddings!")

    st.write("Building FAISS index...")
    faiss_index = faiss.IndexFlatL2(384)
    faiss_index.add(embeddings.astype('float32'))
    st.write("FAISS index built!")

    st.write("Loading GPT-2 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    st.write("Model and tokenizer loaded!")

    return AgentRAG(faiss_index, text_data, embedder, model, tokenizer)

# Streamlit App UI
st.title("AgentRAG: Retrieval-Augmented Generation System")
st.write("Interact with a powerful RAG-based model for knowledge retrieval and response generation.")

agent = initialize_agent()

# Query Input
query = st.text_input("Enter your query:")

if query:
    st.write("**Processing your query...**")

    # Retrieve documents
    st.write("Retrieving relevant documents...")
    relevant_docs = agent.retrieve_documents(query)
    st.write("**Retrieved Documents:**")
    for i, doc in enumerate(relevant_docs, start=1):
        st.write(f"{i}. {doc}")

    # Generate response
    st.write("Generating response...")
    response = agent.generate_response(query, relevant_docs)
    st.write("**Response:**")
    st.write(response)
