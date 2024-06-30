from SupportFunction.BM25Function import find_best_matching_question

# Get the top 5 ranked contexts based on BM25 score for the given question.
question = "bảo hiểm cho người lao động"

scores_word2vec = find_best_matching_question(question)
print(scores_word2vec)
