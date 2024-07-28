import numpy as np

from rank_bm25 import BM25Okapi
from SupportFunction.DataFunction import tokenized_contexts, tokenized_questions, questions, answers

# Tạo BM25 model
bm25_questions = BM25Okapi(tokenized_questions)
bm25_contexts = BM25Okapi(tokenized_contexts)


def get_bm25_scores(user_question, bm25_model):
    bm_25_score = bm25_model.get_scores(user_question)
    scores = (bm_25_score - np.min(bm_25_score)) / \
        (np.max(bm_25_score) - np.min(bm_25_score))
    # Thay thế giá trị nan bằng 0
    scores = np.nan_to_num(scores)
    return scores


def get_top_n_ranked_bm25(question, n=5):
    """
    Returns the top n ranked contexts based on BM25 score.

    Args:
        question (str): The question to rank contexts for.
        n (int): The number of top contexts to return. Default is 5.

    Returns:
        list: A list of tuples containing the context and its BM25 score.
    """
    bm25_scores = get_bm25_scores(question.split(), bm25_questions)
    top_n = np.argsort(bm25_scores)[::-1]
    data = []
    count = 0
    for index in top_n:
        if count == n:
            break
        tmp = {
            'score': f"{bm25_scores[index]:.4f}",
            'question': questions[index],
            'answer': answers[index]
        }
        if tmp in data:
            continue
        data.append(tmp)
        count += 1

    return data
