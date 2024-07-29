import gensim
import numpy as np
from gensim.utils import simple_preprocess

# Load pre-trained Word2Vec model.
word2vec_questions = gensim.models.Word2Vec.load("./models/model_word2vec_question.model")
# word2vec_contexts = gensim.models.Word2Vec.load("./models/word2vec_context.model")


def get_avg_word2vec_vector(user_question, model_word2vec):
    words = user_question
    word_vectors = [model_word2vec.wv[word] for word in words if word in model_word2vec.wv]
    if not word_vectors:
        return np.zeros(model_word2vec.vector_size)
    return np.mean(word_vectors, axis=0)

def get_word2vec_scores(user_question, datas, model_word2vec):
    query_vector = get_avg_word2vec_vector(user_question, model_word2vec)
    scores = []
    for data in datas:
        data_vector = get_avg_word2vec_vector(simple_preprocess(data), model_word2vec)
        score = np.dot(query_vector, data_vector)
        if np.isnan(score):
            score = 0
        scores.append(score)
    return np.array(scores)
    

    