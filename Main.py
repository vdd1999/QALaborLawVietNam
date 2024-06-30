from SupportFunction.BM25Function import get_top_n_ranked_bm25, find_best_matching_question
from SupportFunction.Word2VecFunction import get_word2vec_scores

# Get the top 5 ranked contexts based on BM25 score for the given question.
question = "bảo hiểm y tế"


scores_word2vec = find_best_matching_question(question)
print(scores_word2vec)
