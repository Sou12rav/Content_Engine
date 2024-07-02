import streamlit as st
from langchain import QueryEngine

def chatbot_interface(query_engine, tokenizer, model):
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
