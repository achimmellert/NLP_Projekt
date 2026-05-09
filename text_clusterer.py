import numpy as np
import faiss
import bm25s
import igraph as ig
import leidenalg
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Set, Tuple


class TextClusterer:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base', k_neighbors: int = 3):
        self.model = SentenceTransformer(model_name)
        self.k_neighbors = k_neighbors

  
    def _vectorized_edges(self, labels: np.ndarray) -> Set[Tuple[int, int]]:
        N, k = labels.shape
        sources = np.repeat(np.arange(N), k)
        targets = labels.flatten()
        
        valid = (sources != targets) & (targets >= 0)
        
        edges = np.column_stack((sources[valid], targets[valid]))
        edges.sort(axis=1)
        
        unique_edges = np.unique(edges, axis=0)
        return set(map(tuple, unique_edges))

  
    def _get_faiss_hnsw_edges(self, texts: List[str], num_elements: int) -> Set[Tuple[int, int]]:
        formatted_texts = ["passage: " + t for t in texts]
        embeddings = self.model.encode(formatted_texts, normalize_embeddings=True)
        
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)
        
        k_search = min(self.k_neighbors + 1, num_elements)
        distances, labels = index.search(embeddings, k_search)
        
        return self._vectorized_edges(labels)

  
    def _get_bm25s_edges(self, texts: List[str], num_elements: int) -> Set[Tuple[int, int]]:
        corpus_tokens = bm25s.tokenize(texts)
        
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        k_search = min(self.k_neighbors + 1, num_elements)
        results, scores = retriever.retrieve(corpus_tokens, k=k_search)
        
        return self._vectorized_edges(np.asarray(results, dtype=int))

  
    def fit_predict(self, texts: List[str]) -> pd.DataFrame:
        num_elements = len(texts)
        if num_elements == 0:
            return pd.DataFrame(columns=['text', 'cluster_id'])
        if num_elements == 1:
            return pd.DataFrame({'text': texts, 'cluster_id': [0]})

        faiss_edges = self._get_faiss_hnsw_edges(texts, num_elements)
        bm25_edges = self._get_bm25s_edges(texts, num_elements)
        
        union_edges = faiss_edges.union(bm25_edges)
        
        g = ig.Graph()
        g.add_vertices(num_elements)
        g.add_edges(list(union_edges))
        
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        
        return pd.DataFrame({
            'text': texts,
            'cluster_id': partition.membership
        })
      
