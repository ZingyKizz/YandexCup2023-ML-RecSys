import os
from lib.data.s3 import read_tag_data, read_track_embeddings
from sklearn.model_selection import train_test_split
from lib.data.dataset import TaggingDataset, Collator
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.const import DEVICE
from lib.training.utils import train_epoch, validate_after_epoch, make_test_predictions
from lib.utils import seed_everything, load_config, make_instance
from lib.training.optimizer import get_grouped_parameters


def main(config_path):
    cfg = load_config(config_path)
    cfg_name = cfg["name"]

    seed_everything(cfg["seed"])

    data = read_tag_data(paths=[os.path.join(cfg["data_path"], "data.zip")])
    df_train, df_val = train_test_split(
        data["train"], test_size=cfg["val_size"], random_state=cfg["seed"]
    )
    df_test = data["test"]
    del data

    track_idx2embeds = read_track_embeddings(
        paths=[
            os.path.join(cfg["data_path"], "track_embeddings", f"dir_00{i}.zip")
            for i in range(1, 9)
        ],
    )
    train_dataset = TaggingDataset(df_train, track_idx2embeds)
    val_dataset = TaggingDataset(df_val, track_idx2embeds)
    test_dataset = TaggingDataset(df_test, track_idx2embeds, testing=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=Collator(max_len=cfg["max_len"]),
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=Collator(max_len=cfg["max_len"], testing=True),
        drop_last=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=Collator(max_len=cfg["max_len"], testing=True),
        drop_last=False,
    )

    model = make_instance(cfg["model"], **cfg["model_params"])
    criterion = make_instance(cfg["criterion"])

    epochs = cfg["n_epochs"]
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    optimizer = make_instance(
        cfg["optimizer"],
        get_grouped_parameters(model, cfg["lr"], cfg["lr_alpha"]),
        **cfg["optimizer_params"]
    )

    if "scheduler" in cfg:
        scheduler = make_instance(
            cfg["scheduler"], optimizer, **cfg["scheduler_params"]
        )
    else:
        scheduler = None

    best_score = cfg["best_score"]
    for epoch in tqdm(range(epochs)):
        train_epoch(model, train_dataloader, criterion, optimizer, scheduler)
        score = validate_after_epoch(model, val_dataloader)
        if score > best_score:
            # best_score = score
            make_test_predictions(
                model,
                test_dataloader,
                path="predictions",
                suffix=f"{cfg_name}__epoch_{epoch}_{score:.5f}",
            )


if __name__ == "__main__":
    main("configs/1.yaml")
