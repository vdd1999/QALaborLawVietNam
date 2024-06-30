import json
import os
import re
from underthesea import word_tokenize
from gensim.utils import simple_preprocess


def clean_text(text):
    """
    Cleans the given text by removing newlines, numeric annotations, special characters, and converting it to lowercase.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def load_all_data(data_path):
    """
    Load all data from JSON files in the specified directory.

    Args:
        data_path (str): The path to the directory containing the JSON files.

    Returns:
        dict: A dictionary containing all the loaded data.

    """
    all_data = []
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data['data'])
    return {'data': all_data}


def preprocess_data(data):
    """
    Preprocesses the given data and extracts contexts, questions, and answers.

    Args:
      data (dict): The input data containing articles, paragraphs, and questions.

    Returns:
      tuple: A tuple containing three lists - contexts, questions, and answers.
        - contexts (list): A list of preprocessed contexts.
        - questions (list): A list of preprocessed questions.
        - answers (list): A list of dictionaries, each containing the preprocessed answer text and its start position.
    """
    contexts = []
    questions = []
    answers = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = clean_text(paragraph['context'])
            for qa in paragraph['qas']:
                question = clean_text(qa['question'])
                # Do ngữ liệu nhiều chỗ define thiếu is_impossible nên sẽ mặc định là False
                is_impossible = qa.get('is_impossible', False)
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
    """
    Tokenizes a list of texts using the word_tokenize function.

    Args:
      texts (list): A list of texts to be tokenized.

    Returns:
      list: A list of tokenized texts.
    """
    return [word_tokenize(text, format="text") for text in texts]


data_path = './data'
squad_data = load_all_data(data_path)
contexts, questions, answers = preprocess_data(squad_data)
tokenized_contexts = tokenize_texts(contexts)
tokenized_questions = tokenize_texts(questions)

split_context_pre_train = [context.split() for context in tokenized_contexts]
split_question_pre_train = [simple_preprocess(
    question) for question in tokenized_questions]
