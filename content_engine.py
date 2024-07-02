import PyPDF2
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import faiss
import langchain
import streamlit as st

pdfs = ['data/alphabet_inc_form_10k.pdf', 'data/tesla_inc_form_10k.pdf', 'data/uber_technologies_inc_form_10k.pdf']
documents = []
for pdf in pdfs:
    with open(pdf, 'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        num_pages = pdf_reader.numPages
        text = ''
        for page in range(num_pages):
            page_obj = pdf_reader.getPage(page)
            text += page_obj.extractText()
        documents.append(text)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

vectors = []
for document in documents:
    inputs = tokenizer(document, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :]
    vectors.append(vector.detach().numpy())

index = faiss.IndexFlatL2(768)
index.add(vectors)

query_engine = langchain.QueryEngine(index)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

st.title("Content Engine Chatbot")
st.write("Ask me a question about the documents!")

question = st.text_input("Question:")
if question:
    inputs = tokenizer(question, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    answer = outputs.logits.argmax(-1)
    documents = query_engine.retrieve(answer)
    st.write("Relevant documents:")
    for document in documents:
        st.write(document)
