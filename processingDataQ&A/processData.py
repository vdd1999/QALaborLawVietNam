import json
import os
import re
from underthesea import word_tokenize


def load_all_squad_data(directory_path):
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data['data'])
    return {'data': all_data}


def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def preprocess_squad_data(data):
    contexts = []
    questions = []
    answers = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = clean_text(paragraph['context'])
            for qa in paragraph['qas']:
                question = clean_text(qa['question'])
                is_impossible = qa.get('is_impossible', False)  #Do ngữ liệu nhiều chỗ define thiếu is_impossible nên sẽ mặc định là False
                if not is_impossible:
                    for answer in qa['answers']:
                        answer_text = clean_text(answer['text'])
                        answer_start = answer['answer_start']
                        contexts.append(context)
                        questions.append(question)
                        answers.append({
                            'text': answer_text,
                            'start': answer_start
                        })
    return contexts, questions, answers

def tokenize_texts(texts):
    return [word_tokenize(text, format="text") for text in texts]

directory_path = './data'
squad_data = load_all_squad_data(directory_path)
contexts, questions, answers = preprocess_squad_data(squad_data)
tokenized_contexts = tokenize_texts(contexts)
tokenized_questions = tokenize_texts(questions)