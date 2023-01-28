from torch.utils.data import Dataset
from . import config
import pandas as pd
import torch

class SentimentAnalysisDataset(Dataset):
    """
    
    """
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        review = " ".join(review.split())
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length = self.max_len
        )

        padding_length = self.max_len - len(inputs["input_ids"])
        input_ids = inputs["input_ids"] + ([0] * padding_length)
        attention_mask = inputs["attention_mask"] + ([0] * padding_length)
        token_type_ids = inputs["token_type_ids"] + ([0] * padding_length)
        

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(self.targets[item], dtype=torch.float)
        }

