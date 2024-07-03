import chromadb

chroma_db = chromadb.ChromaDB("my_vector_store")

chroma_db.add_vectors(vectors)
