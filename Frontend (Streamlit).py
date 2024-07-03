import streamlit as st
from llama_index import LlamaIndex

st.title("Content Engine")
st.write("Welcome to the Content Engine!")

query_input = st.text_input("Enter your query:")

submit_button = st.button("Submit")

def display_results(relevant_docs):
    st.write("Relevant documents:")
    for doc in relevant_docs:
        st.write(doc["text"])

if submit_button:
    relevant_docs = process_query(query_input)
    display_results(relevant_docs)
