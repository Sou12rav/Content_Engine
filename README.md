# Content Engine

This repository contains a Content Engine project that analyzes and compares multiple PDF documents, specifically identifying and highlighting their differences. The system utilizes Retrieval Augmented Generation (RAG) techniques to effectively retrieve, assess, and generate insights from the documents.

## Overview

The Content Engine consists of the following components:

* **Backend**: Built using LlamaIndex, ChromaDB, and a local language model (BERT) to process and analyze the PDF documents.
* **Frontend**: A Streamlit app that provides a user interface for interacting with the system and displaying comparative insights.

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Streamlit app using `streamlit run app.py`.
4. Upload the PDF documents to the `documents/` directory.
5. Interact with the system by entering queries in the Streamlit app.

## Notebooks

The following notebooks are available in this repository:

* `document_processing.ipynb`: Notebook for document processing and vector generation.
* `vector_store_ingestion.ipynb`: Notebook for ingesting vectors into ChromaDB.
* `query_engine_development.ipynb`: Notebook for developing the query engine using LlamaIndex.
* `streamlit_code.ipynb`: Notebook for building the Streamlit app.

.
