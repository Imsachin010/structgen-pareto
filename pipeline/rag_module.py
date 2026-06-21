# pipeline/rag_module.py
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class SimpleRAG:
    """
    A minimal TF-IDF based retriever. Load corpus from data/raw_queries.json or list of texts.
    """
    def __init__(self, corpus_texts=None):
        self.corpus = corpus_texts or []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        if self.corpus:
            self._fit(self.corpus)

    def _fit(self, texts):
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    @classmethod
    def from_file(cls, path="data/raw_queries.json"):
        if not os.path.exists(path):
            return cls([])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [item.get("query", "") for item in data]
        obj = cls(texts)
        obj.corpus = texts
        return obj

    def retrieve(self, query, top_k=3):
        if self.tfidf_matrix is None or len(self.corpus) == 0:
            return []
        q_vec = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[::-1][:top_k]
        return [{"score": float(cosine_similarities[i]), "doc": self.corpus[i]} for i in related_docs_indices]
