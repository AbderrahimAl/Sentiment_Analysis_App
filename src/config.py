from transformers import BertTokenizer


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
RANDOM_SEED = 42
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../models/model.bin"
TRAINING_FILE = "../data/IMDB Dataset.csv"
TOKENIZER = BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lower_case=True
    )
