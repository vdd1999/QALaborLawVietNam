from rank_bm25 import BM25Okapi

# Khởi tạo BM25 với các đoạn văn bản đã được token hóa
bm25 = BM25Okapi([context.split() for context in tokenized_contexts])

# WORD 2 VEC
from gensim.models import Word2Vec

# Huấn luyện mô hình Word2Vec với các đoạn văn bản đã được token hóa
tokenized_corpus = [context.split() for context in tokenized_contexts]
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Lưu mô hình Word2Vec đã huấn luyện
word2vec_model.save("word2vec.model")
