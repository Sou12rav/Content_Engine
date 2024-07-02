import faiss

def create_faiss_index(vectors):
    index = faiss.IndexFlatL2(768)
    index.add(vectors)
    return index
