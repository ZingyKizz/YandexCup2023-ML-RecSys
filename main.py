from lib.data.s3 import S3Client
from settings import (
    S3_BUCKET_NAME,
    S3_REGION_NAME,
    S3_ENDPOINT_URL,
    S3_AWS_ACCESS_KEY_ID,
    S3_AWS_SECRET_ACCESS_KEY,
    S3_KEY_BASE,
)
import os
from lib.data.s3 import read_tag_data, read_track_embeddings
from sklearn.model_selection import train_test_split
from lib.data.dataset import TaggingDataset, Collator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
from lib.const import DEVICE
from lib.model.net.baseline import TransNetwork
from lib.model.utils import train_epoch, validate_after_epoch, make_test_predictions


def main():
    path = (
        "/home/jupyter/mnt/s3/rnd-shared/projects/yandex_cup_2023/ML/RecSys/input_data/"
    )
    data = read_tag_data(paths=[os.path.join(path, "data.zip")])
    df_train, df_val = train_test_split(data["train"], test_size=0.1, random_state=11)
    df_test = data["test"]
    del data
    track_idx2embeds = read_track_embeddings(
        paths=[
            os.path.join(path, "track_embeddings", f"dir_00{i}.zip")
            for i in range(1, 9)
        ],
    )
    train_dataset = TaggingDataset(df_train, track_idx2embeds)
    val_dataset = TaggingDataset(df_val, track_idx2embeds)
    test_dataset = TaggingDataset(df_test, track_idx2embeds, testing=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=Collator(max_len=100),
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=Collator(max_len=100, testing=True),
        drop_last=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=Collator(max_len=100, testing=True),
        drop_last=False,
    )

    model = TransNetwork(
        input_dim=768, hidden_dim=1024, n_encoder_layers=6, attention_heads=6
    )
    criterion = nn.BCEWithLogitsLoss()

    epochs = 50
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_score = 0.25
    for epoch in tqdm(range(epochs)):
        train_epoch(model, train_dataloader, criterion, optimizer)
        score = validate_after_epoch(model, val_dataloader)
        if score > best_score:
            best_score = score
            make_test_predictions(
                model,
                test_dataloader,
                path="predictions",
                suffix=f"epoch_{epoch}_{score:.5f}",
            )


if __name__ == "__main__":
    main()
