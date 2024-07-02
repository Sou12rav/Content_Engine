import PyPDF2
import os
import streamlit as st
from langchain.llm import LLM
from llama_index import LlamaIndex
from pinecone import Pinecone

#... (rest of the code remains the same)
# Step 2: Load PDF documents
pdf_paths = ["Alphabet_10K.pdf", "Tesla_10K.pdf", "Uber_10K.pdf"]

# Step 3: Extract text from PDFs
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for path in pdf_paths:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        texts.append(text)
    return texts

documents_text = extract_text_from_pdfs(pdf_paths)

# Step 4: Create embeddings for document content
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def create_embeddings(documents_text):
    embeddings = []
    for text in documents_text:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy())
    return embeddings

document_embeddings = create_embeddings(documents_text)

# Step 5: Store embeddings in Pinecone vector store
pinecone = Pinecone(index_name="content_engine")

def store_embeddings_in_pinecone(document_embeddings):
    pinecone.create_index()
    for i, embedding in enumerate(document_embeddings):
        pinecone.upsert(i, embedding)

store_embeddings_in_pinecone(document_embeddings)

# Step 6: Configure query engine
from llama_index import QueryEngine

query_engine = QueryEngine(index=pinecone)

# Step 7: Integrate local language model
llm = LLM("distilbert-base-uncased")

# Step 8: Develop chatbot interface using Streamlit
st.title("Content Engine")
st.write("Welcome to the Content Engine!")

query = st.text_input("Enter your query:")

if query:
    # Step 9: Retrieve relevant documents using query engine
    results = query_engine.query(query)
    st.write("Relevant documents:")
    for result in results:
        st.write(f"Document {result.doc_id}: {documents_text[result.doc_id]}")

    # Step 10: Generate insights using local language model
    insights = llm.generate_insights(query, results)
    st.write("Insights:")
    for insight in insights:
        st.write(insight)
