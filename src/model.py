import torch
from torch import nn
from transformers import BertModel
import config

class SentimentClassifier(nn.Module):
    def __init__(self) -> None:
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask, token_type_ids):
        _, out2 = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict=False
        )
        d_out2 = self.bert_drop(out2)
        output = self.out(d_out2)
        return output

