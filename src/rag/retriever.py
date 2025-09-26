from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGRetriever:
    def __init__(self, documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding.astype('float32'), top_k)
        return [self.documents[i] for i in indices[0]]