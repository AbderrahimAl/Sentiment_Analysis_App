import torch
from . import config
from .model import SentimentClassifier
from fastapi import FastAPI

tokenizer = config.TOKENIZER
max_len = config.MAX_LEN

def prepare_input(review, tokenizer=tokenizer, max_len=max_len):
    """
    """
    inputs = tokenizer.encode_plus(
        review,
        max_length=max_len,
        add_special_tokens=True
    )
    padding_length = max_len - len(inputs['input_ids'])
    input_ids = inputs['input_ids'] + ([0] * padding_length)
    attention_mask = inputs['attention_mask'] + ([0] * padding_length)
    token_type_ids = inputs['token_type_ids'] + ([0] * padding_length)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    return input_ids, attention_mask, token_type_ids

def load_model():
    model = SentimentClassifier()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu'))())
    return model

app = FastAPI()
@app.get("/predict/{review}")
def predict(review: str):
    input_ids, attention_mask, token_type_ids = prepare_input(review)
    model = load_model()
    raw_output = model(
        input_ids,
        attention_mask,
        token_type_ids
    )
    prediction = torch.round(torch.sigmoid(raw_output)).detach().numpy()[0]

    sentiement_pred = 'positive' if prediction == 1 else 'negative'
    return {
        "review": review, 
        "sentiment": sentiement_pred
        }
