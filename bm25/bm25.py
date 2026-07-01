import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([], dtype=np.float64)
        
    doc_lens = [len(doc) for doc in docs]
    avgdl = sum(doc_lens) / N
    
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
            
    scores = []
    for doc, d_len in zip(docs, doc_lens):
        score = 0.0
        doc_counts = Counter(doc)
        
        for term in query_tokens:
            if term not in doc_counts:
                continue
                
            tf = doc_counts[term]
          
            idf = math.log(1.0 + (N - df[term] + 0.5) / (df[term] + 0.5))
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1.0 - b + b * (d_len / avgdl)) if avgdl > 0 else tf + k1
            
            score += idf * (numerator / denominator)
            
        scores.append(score)
        
    return np.array(scores, dtype=np.float64)