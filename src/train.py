import pandas as pd
import dataset
from . import config
from . import engine
import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import SentimentClassifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def trainer():

    # load the data
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")[:100]
    le = LabelEncoder()
    df.sentiment = le.fit(df.sentiment).transform(df.sentiment)
    df_train, df_valid = train_test_split(
        df, 
        train_size=0.9,
        random_state=config.RANDOM_SEED,
        stratify=df.sentiment.values)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    train_dataset = dataset.SentimentAnalysisDataset(
        reviews=df_train.review.values,
        targets=df_train.sentiment.values
    )
    valid_dataset = dataset.SentimentAnalysisDataset(
        reviews=df_valid.review.values,
        targets=df_valid.sentiment.values
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE
    )
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.VALID_BATCH_SIZE
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentimentClassifier().to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(
        params=optimizer_parameters,
        lr=3e-5
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
            train_data_loader,
            model,
            optimizer,
            scheduler,
            device
        )
        outputs, targets = engine.eval_fn(
                                valid_data_loader,
                                model,
                                device
                            )
        outputs = (np.array(outputs) >= 0.5) * 1
        accuracy = accuracy_score(targets, outputs)
        print(f"epoch {epoch}")
        print(f"Accuracy score {accuracy}")
        print("------------------------------")
        if accuracy > best_accuracy:
            torch.save(model.state_dict, config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    trainer()
