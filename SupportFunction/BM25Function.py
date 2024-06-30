import numpy as np

from rank_bm25 import BM25Okapi
from SupportFunction.DataFunction import split_question_pre_train, questions, tokenize_texts
from SupportFunction.Word2VecFunction import get_word2vec_scores
from gensim.utils import simple_preprocess

# Tạo BM25 model
bm25 = BM25Okapi(split_question_pre_train)


def get_bm25_score(question):
    tokenized_query = simple_preprocess(tokenize_texts([question])[0])
    bm_25_score = bm25.get_scores(tokenized_query)
    scores = (bm_25_score - np.min(bm_25_score)) / \
        (np.max(bm_25_score) - np.min(bm_25_score))
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
    bm25_scores = get_bm25_score(question)
    top_n = np.argsort(bm25_scores)[::-1][:n]
    data = []
    for index in top_n:
        tmp = {
            'score': f"{bm25_scores[index]:.4f}",
            'question': questions[index],
        }
        if tmp in data:
            continue
        data.append(tmp)

    return data


def find_best_matching_question(user_question):
    bm25_scores = get_bm25_score(user_question)
    word2vec_scores = get_word2vec_scores(user_question)

    combined_scores = word2vec_scores + bm25_scores
    combined_scores = np.nan_to_num(combined_scores)

    if np.all(combined_scores == 0):
        return "Không có câu hỏi được tìm thấy"

    best_idx = np.argmax(combined_scores)
    best_question = questions[best_idx]
    return best_question
