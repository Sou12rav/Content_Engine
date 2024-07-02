Content Engine Project 
Overview
This repo holds the code for the Content Engine project. We built a chatbot that grabs and compares info from multiple PDFs. Our bot uses NLP and info retrieval tricks to give good answers to what users ask. Architecture Our system has these parts:
Document Parser: This bit takes PDFs and pulls out the words.
Vector Generator: This part uses DistilBERT (a pre-trained language model) to turn text into number vectors. 
Vector Store: This bit keeps the document vectors in a Faiss index. Faiss makes finding similar stuff quick. 
Query Engine: This part uses the Faiss index to find docs that match what the user wants. 

working:
The user inputs a query into the chatbot interface.
The query is tokenized and fed into the DistilBERT model, which generates a vector representation of the query.
The vector representation of the query is used to search the Faiss index, which returns a list of documents that are similar to the query.
The chatbot interface displays the relevant documents to the user.
Components

The repository contains the following components:
content_engine.py: A Python script containing the code for parsing documents, generating vectors, storing in vector store, configuring query engine, integrating LLM, and developing chatbot interface.
langchain_config.json: A configuration file for LangChain, which is used to configure the Faiss index.
faiss_index.py: A Python script for creating and managing the Faiss index.
distilbert_model.py: A Python script for loading and using the DistilBERT model.
streamlit_app.py: A Python script for developing the chatbot interface using Streamlit.
