import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

class BasicRAG:
    def __init__(self, faiss_index, embedder, model, tokenizer):
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.model = model
        self.tokenizer = tokenizer

    def retrieve_documents(self, query, k=3):
        query_embedding = self.embedder.encode([query]).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)
        return [query]  # Since we index only the query, retrieval returns the query itself

    def generate_response(self, query, relevant_docs):
        context = " ".join(relevant_docs[:2]) + " " + query
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=800,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def handle_query(self, query, k=3):
        retrieved_docs = self.retrieve_documents(query, k)
        response = self.generate_response(query, retrieved_docs)
        return response

    def calculate_perplexity(self, query, max_length=1024):
        print("Calculating perplexity...")
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            log_probs = -outputs.loss.item() * input_ids.size(1)
        perplexity = math.exp(-log_probs / input_ids.size(1))
        print("Done: Perplexity calculated.")
        return perplexity

# Example initialization and usage
print("Loading model, tokenizer, and FAISS index...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index with appropriate dimensions
faiss_index = faiss.IndexFlatL2(384)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.eval()

print("Initializing BasicRAG...")
rag = BasicRAG(faiss_index, embedder, model, tokenizer)
print("BasicRAG ready!")

# Query and FAISS indexing
query = "What causes cancer?"
query_embedding = embedder.encode([query]).astype('float32')
faiss_index.add(query_embedding)

# Generate a response
response = rag.handle_query(query)
print('Query:', query)
print('Response:', response)

# Perplexity evaluation using the query
print("Evaluating perplexity on the query...")
perplexity = rag.calculate_perplexity(query)
print(f"Perplexity: {perplexity}")
