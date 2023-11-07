import datetime as dt
import torch
import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd
import os
from torch_ema import ExponentialMovingAverage

from lib.const import DEVICE
from lib.utils import make_instance
from lib.training.optimizer import get_grouped_parameters


def batch_to_device(embeds):
    if isinstance(embeds, tuple):
        return [x.to(DEVICE) for x in embeds]
    return embeds.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer, scheduler=None, ema=None):
    model.train()
    running_loss = None
    alpha = 0.8
    for iteration, data in enumerate(loader):
        optimizer.zero_grad()
        track_idxs, x, target = data
        x = batch_to_device(x)
        target = target.to(DEVICE)
        pred_logits = model(*x)
        ce_loss = criterion(pred_logits, target)
        ce_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()
        if running_loss is None:
            running_loss = ce_loss.item()
        else:
            running_loss = alpha * running_loss + (1 - alpha) * ce_loss.item()
        if iteration % 100 == 0:
            print(
                "   {} batch {} loss {}".format(
                    dt.datetime.now(), iteration + 1, running_loss
                )
            )


@torch.no_grad()
def predict(model, loader, ema=None):
    model.eval()
    if ema is not None:
        ema.store()
        ema.copy_to()
    track_idxs = []
    predictions = []
    for data in loader:
        track_idx, x, _ = data
        x = batch_to_device(x)
        pred_logits = model(*x)
        pred_probs = torch.sigmoid(pred_logits)
        predictions.append(pred_probs.cpu().detach().numpy())
        track_idxs.append(track_idx.cpu().detach().numpy())
    if ema is not None:
        ema.restore()
    predictions = np.vstack(predictions)
    track_idxs = np.vstack(track_idxs).ravel()
    return track_idxs, predictions


def validate_after_epoch(model, loader, ema=None):
    ys_true = {x[0]: x[-1] for x in loader.dataset}
    track_idxs, predictions = predict(model, loader, ema=ema)
    yts, yps = [], []
    for tid, y_pred in zip(track_idxs, predictions):
        yts.append(ys_true[tid])
        yps.append(y_pred)
    score = average_precision_score(yts, yps)
    print(f"AveragePrecision: {score}")
    return score


def make_test_predictions(model, test_dataloader, path=None, suffix=None, ema=None):
    track_idxs, predictions = predict(model, test_dataloader, ema=ema)
    predictions_df = pd.DataFrame(
        [
            {"track": track, "prediction": ",".join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ]
    )
    if suffix is not None:
        name = f"prediction__{suffix}.csv"
    else:
        name = "prediction.csv"
    if path is not None:
        os.makedirs(path, exist_ok=True)
        name = os.path.join(path, name)
    predictions_df.to_csv(name, index=False)
    return predictions


def make_val_test_predictions(
    model, val_dataloader, test_dataloader, cfg_name, fold_idx, epoch, score, ema=None
):
    make_test_predictions(
        model,
        val_dataloader,
        path="predictions_val",
        suffix=f"cfg={cfg_name}__fold_idx={fold_idx}__epoch={epoch}__score={score:.5f}",
        ema=ema,
    )
    make_test_predictions(
        model,
        test_dataloader,
        path="predictions_test",
        suffix=f"cfg={cfg_name}__fold_idx={fold_idx}__epoch={epoch}__score={score:.5f}",
        ema=ema,
    )


def init_nn_stuff(cfg):
    model = make_instance(cfg["model"], **cfg["model_params"])
    criterion = make_instance(cfg["criterion"], **cfg.get("criterion_params", {}))
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    optimizer = make_instance(
        cfg["optimizer"],
        get_grouped_parameters(model, cfg["lr"], cfg["lr_alpha"]),
        **cfg.get("optimizer_params", {}),
    )
    if "scheduler" in cfg:
        scheduler = make_instance(
            cfg["scheduler"], optimizer, **cfg.get("scheduler_params", {})
        )
    else:
        scheduler = None
    if cfg.get("use_ema", False):
        ema = ExponentialMovingAverage(
            model.parameters(), decay=cfg.get("ema_decay", 0.995)
        )
    else:
        ema = None
    return model, criterion, optimizer, scheduler, ema
