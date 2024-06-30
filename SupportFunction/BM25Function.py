import numpy as np

from rank_bm25 import BM25Okapi
from SupportFunction.DataFunction import split_question_pre_train, questions
from SupportFunction.Word2VecFunction import get_word2vec_scores
from gensim.utils import simple_preprocess

# Tạo BM25 model
bm25 = BM25Okapi(split_question_pre_train)


def get_bm25_score(question):
    tokenized_query = simple_preprocess(question)
    return bm25.get_scores(tokenized_query)


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
        data.append({
            'score': f"{bm25_scores[index]:.4f}",
            'question': questions[index],
        })

    return data


def find_best_matching_question(user_question):
    bm25_scores = get_bm25_score(user_question)
    print('BM25 scores:', bm25_scores)
    word2vec_scores = get_word2vec_scores(user_question)
    print('Word2Vec scores:', word2vec_scores)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / \
        (np.max(bm25_scores) - np.min(bm25_scores))
    word2vec_scores = (word2vec_scores - np.min(word2vec_scores)) / \
        (np.max(word2vec_scores) - np.min(word2vec_scores))

    # Combine scores with equal weight
    if bm25_scores.size != word2vec_scores.size:
        combined_scores = bm25_scores
    else:
        combined_scores = bm25_scores + word2vec_scores

    best_idx = np.argmax(combined_scores)
    best_question = questions[best_idx]
    print('Best matching question:', best_idx)
    return best_question
