import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_path = './models/phoBert_model'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)


def check_model_exist():
    """
    Check if the model exists

    Args:
        model_path (str): The path to the model.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    global model_path
    return os.path.exists(model_path)
