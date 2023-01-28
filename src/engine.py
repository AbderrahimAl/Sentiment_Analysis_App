from .model import SentimentClassifier
from tqdm import tqdm
import torch
from torch import nn

def loss_fn(y_pred, y):
    return nn.BCEWithLogitsLoss()(y_pred, y.view(-1, 1))


def train_fn(data_loader, model, optimizer, scheduler, device):
    """
    """
    model.train()
    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids = data["input_ids"].to(device, torch.long)
        attention_mask = data["attention_mask"].to(device, torch.long)
        token_type_ids = data["token_type_ids"].to(device, torch.long)
        targets = data["target"].to(device, torch.float)
        outputs = model(
            input_ids, 
            attention_mask, 
            token_type_ids)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    """
    """
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.inference_mode():
        for batch_id, data in tqdm(enumerate(data_loader), total= len(data_loader)):
            input_ids = data["input_ids"].to(device, torch.long)
            attention_mask = data["attention_mask"].to(device, torch.long)
            token_type_ids = data["token_type_ids"].to(device, torch.long)
            targets = data["target"].to(device, torch.float)
            outputs = model(
                input_ids,
                attention_mask,
                token_type_ids
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets       

