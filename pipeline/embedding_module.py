from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRetriever:
    def __init__(self, documents, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        ranked_idx = np.argsort(similarities)[::-1][:top_k]

        return [
            {"doc": self.documents[i], "score": float(similarities[i])}
            for i in ranked_idx
        ]
