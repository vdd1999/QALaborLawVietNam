from rank_bm25 import BM25Okapi

# Tạo BM25 model
bm25 = BM25Okapi(split_question_pre_train)

# Changed
def get_avg_word2vec_vector(user_question):
    words = user_question
    word_vectors = [model_word2vec.wv[word] for word in words if word in model_word2vec.wv]
    if not word_vectors:
        return np.zeros(model_word2vec.vector_size)
    return np.mean(word_vectors, axis=0)

# Changed
def get_word2vec_scores(user_question, questions):
    query_vector = get_avg_word2vec_vector(user_question)
    scores = []
    for question in questions:
        question_vector = get_avg_word2vec_vector(question)
        score = np.dot(query_vector, question_vector)
        if np.isnan(score):
            score = 0
        scores.append(score)
    return np.array(scores)

# Added
def get_bm25_scores(user_question):
    bm_25_score = bm25.get_scores(user_question)
    scores = (bm_25_score - np.min(bm25_scores)) / (np.max(bm_25_score) - np.min(bm_25_score))
    # Thay thế giá trị nan bằng 0
    scores = np.nan_to_num(scores)
    return scores

# Added
def get_combined_scores(user_question, questions):
    tokenized_query = simple_preprocess(tokenize_texts([user_question])[0])
    word2vec_scores = get_word2vec_scores(tokenized_query, questions)
    bm25_scores = get_bm25_scores(tokenized_query)
    word2vec_scores = (word2vec_scores - np.min(word2vec_scores)) / (np.max(word2vec_scores) - np.min(word2vec_scores))
    combined_scores = word2vec_scores + bm25_scores
    # Thay thế giá trị nan bằng 0 trong combined_scores
    combined_scores = np.nan_to_num(combined_scores)
    return combined_scores

def find_best_matching_question(user_question, questions):
    combined_scores = get_combined_scores(user_question, questions)
    if np.all(combined_scores == 0):
        return "Không có câu hỏi được tìm thấy"
    best_match_idx = np.argmax(combined_scores)
    return questions[best_match_idx]
    
input_question = "người lao động quyền lợi gì khi tham gia bảo hiểm y tế"
best_question = find_best_matching_question(input_question, questions)

print(f"User Question: {input_question}")
print(f"Best matching question: {best_question}")