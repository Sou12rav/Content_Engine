import os
import json
from llama_index import LlamaIndex, VectorStore
from transformers import AutoModelForSequenceClassification, AutoTokenizer

pdf_docs = [
    "Documents/goog-10-k-2023.pdf",
    "Documents/tsla-20231231-gen.pdf",
    "Documents/uber-10-k-2023.pdf"
]

def parse_documents(pdf_docs):
    documents = []
    for pdf_doc in pdf_docs:
        with open(pdf_doc, "rb") as f:
            pdf_content = f.read()
        text, structure = parse_pdf(pdf_content)
        documents.append({"text": text, "structure": structure})
    return documents

def generate_vectors(documents):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    vectors = []
    for document in documents:
        inputs = tokenizer(document["text"], return_tensors="pt")
        outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :]
        vectors.append(vector.detach().numpy())
    return vectors

vector_store = VectorStore("chromadb", "my_vector_store")
vector_store.add_vectors(vectors)

llama_index = LlamaIndex(vector_store, "my_llama_index")
llama_index.configure_query_engine()

llm = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

def process_query(query):
  
    inputs = tokenizer(query, return_tensors="pt")
    outputs = llm(**inputs)
    insights = outputs.last_hidden_state[:, 0, :]
    relevant_docs = llama_index.query(insights)
    return relevant_docs
