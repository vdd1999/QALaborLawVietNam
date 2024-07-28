import json
import re
from underthesea import word_tokenize


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


def load_all_data(input_file: str) -> dict:
    """
    Loads the data from the given input file.

    Args:
        input_file (str): The path to the input file containing the data.

    Returns:
        dict: The loaded data as a dictionary
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


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
        - context_raws (list): A list of raw contexts.
        - question_raws (list): A list of raw questions. 
    """
    contexts = []
    context_raws = []
    questions = []
    answers = []
    question_raws = []
    for dataSquad in data:
        for article in dataSquad['data']:
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
                            context_raws.append(paragraph['context'])
                            questions.append(question)
                            question_raws.append(qa['question'])
                            answers.append({
                                'text': answer_text,
                                'start': answer_start
                            })
    return contexts, questions, answers, context_raws, question_raws


def tokenize_texts(texts):
    """
    Tokenizes a list of texts using the word_tokenize function.

    Args:
      texts (list): A list of texts to be tokenized.

    Returns:
      list: A list of tokenized texts.
    """
    return [text.split() for text in texts]


data_path = './data/qa_train.json'
squad_data = load_all_data(data_path)
contexts, questions, answers, context_raws, question_raws = preprocess_data(
    squad_data)
tokenized_contexts = tokenize_texts(contexts)
tokenized_questions = tokenize_texts(questions)
