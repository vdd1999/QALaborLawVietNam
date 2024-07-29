import numpy as np
import torch

from SupportFunction.BM25Function import bm25_questions, get_bm25_scores
from SupportFunction.Word2VecFunction import get_word2vec_scores, word2vec_questions
from SupportFunction.DataFunction import questions, contexts, answers, context_raws, question_raws
from SupportFunction.PhoBertFunction import model, tokenizer, check_model_exist
from collections import Counter


def get_combined_scores(user_question, datas, model_word2vec, bm25_model):
    tokenized_query = user_question.split()
    word2vec_scores = get_word2vec_scores(
        tokenized_query, datas, model_word2vec)
    bm25_scores = get_bm25_scores(tokenized_query, bm25_model)
    word2vec_scores = (word2vec_scores - np.min(word2vec_scores)) / \
        (np.max(word2vec_scores) - np.min(word2vec_scores))
    combined_scores = word2vec_scores + bm25_scores
    # Thay thế giá trị nan bằng 0 trong combined_scores
    combined_scores = np.nan_to_num(combined_scores)
    return combined_scores


def get_combined_bm25_word2vec_scores(user_question):
    tokenized_query = user_question.split()
    word2vec_scores = get_word2vec_scores(
        tokenized_query, questions, word2vec_questions)
    bm25_scores = get_bm25_scores(tokenized_query, bm25_questions)
    word2vec_scores = (word2vec_scores - np.min(word2vec_scores)) / \
        (np.max(word2vec_scores) - np.min(word2vec_scores))
    combined_scores = word2vec_scores + bm25_scores
    # Thay thế giá trị nan bằng 0 trong combined_scores
    combined_scores = np.nan_to_num(combined_scores)
    if np.all(combined_scores == 0):
        return "Không có câu hỏi được tìm thấy", {'text': "Không có câu trả lời được tìm thấy"}

    best_match_idx = np.argmax(combined_scores)
    best_question = questions[best_match_idx]
    best_ans = answers[best_match_idx]
    print(best_question, best_ans)
    return best_question, best_ans


def find_best_matching_question(user_question, questions, context_raws, question_raws):
    combined_scores = get_combined_scores(
        user_question, questions, word2vec_questions, bm25_questions)
    if np.all(combined_scores == 0):
        return "", "", "Không có câu hỏi được tìm thấy"
    best_match_idx = np.argmax(combined_scores)
    return question_raws[best_match_idx], context_raws[best_match_idx], answers[best_match_idx]


def get_answer(question):
    if not check_model_exist():
        return "Vui lòng copy model phoBert vào thư mục models/phoBert_model"
    bestQuestion, contextMatch, answerMatch = find_best_matching_question(
        question, questions, context_raws, question_raws)
    inputs = tokenizer(bestQuestion.lower(), contextMatch, return_tensors="pt",
                       max_length=258, padding="max_length", truncation="only_second")
    try:
      with torch.no_grad():
          outputs = model(**inputs)
      answer_start_index = outputs.start_logits.argmax()
      answer_end_index = outputs.end_logits.argmax()
      predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    
      answer_predict = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
      predict_score = f1_score(answer_predict, answerMatch['text'])
      answer_match_score = f1_score(answerMatch['text'], answerMatch['text'])
      if(answer_predict == ""):
          answer_result = answerMatch['text']
      elif predict_score > answer_match_score:
        answer_result = answer_predict
      else:
          answer_result = answer_predict
    except IndexError as e:
        print(f"Lỗi: {e}")
        answer_result = "Không tìm thấy câu trả lời phù hợp"
    return answer_result


def get_predicted_answer(question):
    if not check_model_exist():
        return "Vui lòng copy model phoBert vào thư mục models/phoBert_model"
    best_question, match_context, ans = find_best_matching_question(
        question, questions, context_raws, question_raws)

    inputs = tokenizer.encode_plus(
        best_question, match_context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer



def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
