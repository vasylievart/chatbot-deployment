import random
import json
import torch
import os

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

def load_model_data(lang='en'):
    # Correct paths based on folder structure
    base_path = os.path.join("data", lang)
    intent_path = os.path.join(base_path, "intents.json")
    model_path = os.path.join(base_path, f"data_{lang}.pth")

    # Load intents
    with open(intent_path, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    # Load model
    data = torch.load(model_path)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return model, all_words, tags, intents


def get_response(msg, lang='en'):
    model, all_words, tags, intents = load_model_data(lang)

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return "Lo siento, no entendí tu mensaje. Puedes contactnos por WhatsApp or email."

