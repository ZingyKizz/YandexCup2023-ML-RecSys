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
from lib.data.dataset import TaggingDataset, collate_fn, collate_fn_test
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
from lib.const import DEVICE
from lib.model.net.baseline import Network
from lib.model.utils import train_epoch, validate_after_epoch, make_test_predictions


def main():
    # s3 = S3Client(
    #     endpoint_url=S3_ENDPOINT_URL,
    #     region_name=S3_REGION_NAME,
    #     aws_access_key_id=S3_AWS_ACCESS_KEY_ID,
    #     aws_secret_access_key=S3_AWS_SECRET_ACCESS_KEY,
    # )
    path = (
        "/Users/yaroslav.hnykov/Desktop/Study/VCS/YandexCUP2023/ML/RecSys/input_data/"
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
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_test
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_test
    )
    model = Network(input_dim=768, hidden_dim=768)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 20
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in tqdm(range(epochs)):
        train_epoch(model, train_dataloader, criterion, optimizer)
        score = validate_after_epoch(model, val_dataloader)
    make_test_predictions(
        model, test_dataloader, path="predictions", suffix=f"{score:.5f}"
    )


if __name__ == "__main__":
    main()
