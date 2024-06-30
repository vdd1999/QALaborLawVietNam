import gensim
import numpy as np
from gensim.utils import simple_preprocess
from SupportFunction.DataFunction import questions, tokenize_texts

# Load pre-trained Word2Vec model.
model_word2vec = gensim.models.Word2Vec.load("./models/word2vec.model")


def get_vector_word(word):
    """
    Retrieves the vector representation of a word from the Word2Vec model.

    Args:
        word (str): The word to retrieve the vector for.

    Returns:
        numpy.ndarray or None: The vector representation of the word if it exists in the Word2Vec model,
        otherwise returns None.
    """
    word = word.lower()
    if word in model_word2vec.wv:
        return model_word2vec.wv[word]
    else:
        return None


def similarity_word(word, top=5):
    """
    Returns a list of similar words to the given word.

    Parameters:
    word (str): The word to find similar words for.
    top (int): The number of similar words to return. Default is 5.

    Returns:
    list: A list of tuples containing the similar words and their similarity scores.
    """
    word = word.lower()
    try:
        similiars = model_word2vec.wv.most_similar(word, topn=top)
        return similiars
    except:
        return []


def get_avg_word2vec_vector(question):
    words = simple_preprocess(tokenize_texts([question])[0])
    vector = np.mean([model_word2vec.wv[word]
                     for word in words if word in model_word2vec.wv], axis=0)
    return vector


def get_word2vec_scores(user_question):
    query_vector = get_avg_word2vec_vector(user_question)
    scores = []
    for question in questions:
        question_vector = get_avg_word2vec_vector(question)
        score = np.dot(query_vector, question_vector)
        if np.isnan(score):
            score = 0
        scores.append(score)
    return np.array(scores)
