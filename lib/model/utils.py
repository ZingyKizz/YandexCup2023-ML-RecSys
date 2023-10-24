import datetime as dt
import torch
import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd
import os

from lib.const import DEVICE


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = None
    alpha = 0.8
    for iteration, data in enumerate(loader):
        optimizer.zero_grad()
        track_idxs, embeds, target = data
        embeds = [x.to(DEVICE) for x in embeds]
        target = target.to(DEVICE)
        pred_logits = model(embeds)
        ce_loss = criterion(pred_logits, target)
        ce_loss.backward()
        optimizer.step()

        if running_loss is None:
            running_loss = ce_loss.item()
        else:
            running_loss = alpha * ce_loss.item() + (1 - alpha) * ce_loss.item()
        if iteration % 100 == 0:
            print(
                "   {} batch {} loss {}".format(
                    dt.datetime.now(), iteration + 1, running_loss
                )
            )


torch.no_grad()


def predict(model, loader):
    model.eval()
    track_idxs = []
    predictions = []
    for data in loader:
        track_idx, embeds = data
        embeds = [x.to(DEVICE) for x in embeds]
        pred_logits = model(embeds)
        pred_probs = torch.sigmoid(pred_logits)
        predictions.append(pred_probs.cpu().detach().numpy())
        track_idxs.append(track_idx.cpu().detach().numpy())
    predictions = np.vstack(predictions)
    track_idxs = np.vstack(track_idxs).ravel()
    return track_idxs, predictions


def validate_after_epoch(model, loader):
    ys_true = {x[0]: x[-1] for x in loader.dataset}
    track_idxs, predictions = predict(model, loader)
    yts, yps = [], []
    for tid, y_pred in zip(track_idxs, predictions):
        yts.append(ys_true[tid])
        yps.append(y_pred)
    score = average_precision_score(yts, yps)
    print(f"AveragePrecision: {score}")
    return score


def make_test_predictions(model, test_dataloader, path=None, suffix=None):
    track_idxs, predictions = predict(model, test_dataloader)
    predictions_df = pd.DataFrame(
        [
            {"track": track, "prediction": ",".join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ]
    )
    if suffix is not None:
        name = f"prediction_{suffix}.csv"
    else:
        name = "prediction.csv"
    if path is not None:
        os.makedirs(path, exist_ok=True)
        name = os.path.join(path, name)
    predictions_df.to_csv(name, index=False)
