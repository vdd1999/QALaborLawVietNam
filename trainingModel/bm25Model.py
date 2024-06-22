from rank_bm25 import BM25Okapi

# Tạo BM25 model
bm25 = BM25Okapi(split_question_pre_train)

def get_avg_word2vec_vector(question):
    words = simple_preprocess(question)
    vector = np.mean([model_word2vec.wv[word] for word in words if word in model_word2vec.wv], axis=0)
    return vector

def get_word2vec_scores(user_question, questions):
    query_vector = get_avg_word2vec_vector(user_question)
    scores = []
    for question in questions:
        question_vector = get_avg_word2vec_vector(question)
        score = np.dot(query_vector, question_vector)
        scores.append(score)
    return np.array(scores)

def find_best_matching_question(user_question, questions):
    tokenized_query = simple_preprocess(user_question)
    bm25_scores = bm25.get_scores(tokenized_query)
    word2vec_scores = get_word2vec_scores(user_question, questions)
    
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    word2vec_scores = (word2vec_scores - np.min(word2vec_scores)) / (np.max(word2vec_scores) - np.min(word2vec_scores))
    
    # Combine scores with equal weight
    combined_scores = bm25_scores + word2vec_scores
    
    best_idx = np.argmax(combined_scores)
    best_question = questions[best_idx]
    return best_question
input_question = "Lương ngoài giờ tính như thế nào"
best_question = find_best_matching_question(input_question, questions)

print(f"User Question: {input_question}")
print(f"Best matching question: {best_question}")