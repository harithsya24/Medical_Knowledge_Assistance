import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import math
import requests

# Load the dataset
print("Loading the dataset...")
dataset = load_dataset("MedRAG/textbooks", split="train")
text_data = [item['content'] for item in dataset]
print("Done: Data loading!")

# Encode the dataset using SentenceTransformers
print("Encoding the dataset with SentenceTransformers...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(text_data)
print(f"Done: Encoded {len(embeddings)} embeddings!")

# Build FAISS index
print("Building the FAISS index...")
faiss_index = faiss.IndexFlatL2(384)
faiss_index.add(embeddings.astype('float32'))
print("Done: FAISS index built!")

# Load a generative GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.eval()
print("Done: GPT-2 model and tokenizer loaded!")

import requests

class AgentRAG:
    def __init__(self, faiss_index, text_data, embedder, model, tokenizer, external_tools=None):
        self.faiss_index = faiss_index
        self.text_data = text_data
        self.embedder = embedder
        self.model = model
        self.tokenizer = tokenizer
        self.external_tools = external_tools if external_tools is not None else []

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

    def handle_query(self, query, k=3):
        retrieved_docs = self.retrieve_documents(query, k)
        action_required = self.analyze_query(query, retrieved_docs)

        if action_required:
            additional_info = self.execute_agent_action(query)
            context = " ".join(retrieved_docs) + " " + query + " " + additional_info
            response = self.generate_response(query, [context])
        else:
            response = self.generate_response(query, retrieved_docs)

        return response

    def analyze_query(self, query, relevant_docs):
        if "side effects" in query.lower():
            return True
        return False

    def execute_agent_action(self, query):
        if self.external_tools:
            for tool in self.external_tools:
                if tool["type"] == "database":
                    return self.query_external_database(query, tool)
        return "No additional information found."

    def query_external_database(self, query, tool):
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': 5,
            'api_key': tool["details"]["api_key"],
            'retmode': 'xml'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return self.parse_pubmed_results(response.text)
        else:
            return f"Error: Unable to query external database (Status code: {response.status_code})"

    def parse_pubmed_results(self, xml_data):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_data)
        article_ids = [id.text for id in root.findall(".//Id")]
        return f"Found articles with IDs: {', '.join(article_ids)}"

# Perplexity Calculation Function
def calculate_perplexity(texts, tokenizer, model, max_length=1024):
    print("Calculating perplexity...")
    total_log_prob = 0
    total_tokens = 0

    for idx, text in enumerate(texts):
        print(f"Processing text {idx + 1}/{len(texts)}...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            log_probs = -outputs.loss.item() * input_ids.size(1)

        total_log_prob += log_probs
        total_tokens += input_ids.size(1)

    perplexity = math.exp(-total_log_prob / total_tokens)
    print("Done: Perplexity calculated.")
    return perplexity

external_tools = [
    {"type": "database", "name": "PubMed", "details": {"api_key": "690ef31d7799a92bd003887e68ebf0242208y"}}
]

# Create an instance of the agent
print("Initializing the AgentRAG...")
agent = AgentRAG(faiss_index, text_data, embedder, model, tokenizer, external_tools=external_tools)
print("Done: AgentRAG initialized!")

# Query Handling
query = "What are the side effects of aspirin?"
response = agent.handle_query(query)
print('Query:',query)
print(response)

# Perplexity Evaluation on Dataset
print("Evaluating perplexity on the first 10 texts of the dataset...")
ppl = calculate_perplexity(text_data[:50], tokenizer, model)
print(f"Perplexity on Dataset: {ppl}")