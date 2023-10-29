import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from lib.const import NUM_TAGS


class TaggingDataset(Dataset):
    def __init__(self, df, track_idx2embeds, testing=False):
        self.df = df
        self.track_idx2embeds = track_idx2embeds
        self.testing = testing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track
        embeds = self.track_idx2embeds[track_idx]
        target = -1
        if self.testing:
            return track_idx, embeds, target
        tags = [int(x) for x in row.tags.split(",")]
        target = np.zeros(NUM_TAGS)
        target[tags] = 1
        return track_idx, embeds, target


class Collator:
    def __init__(self, max_len=None):
        self.max_len = max_len

    def __call__(self, b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
        embeds = [torch.from_numpy(x[1][: self.max_len]) for x in b]
        attention_mask = self._create_attention_mask(embeds)
        embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
        targets = np.vstack([x[2] for x in b])
        targets = torch.from_numpy(targets)
        return track_idxs, (embeds, attention_mask), targets

    @staticmethod
    def _create_attention_mask(embeds):
        lens = torch.tensor([len(e) for e in embeds])
        attention_mask = (torch.arange(max(lens))[None, :] < lens[:, None]).float()
        return attention_mask


def make_dataloader(df, track_idx2embeds, cfg, testing=False):
    dataset = TaggingDataset(df, track_idx2embeds, testing=testing)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=not testing,
        collate_fn=Collator(max_len=cfg["max_len"]),
        drop_last=False,
    )
    return dataloader


def cross_val_split(df, track_idx2embeds, cfg):
    for train_indices, val_indices in KFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"]
    ).split(df):
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        yield (
            make_dataloader(
                train_df,
                track_idx2embeds,
                cfg,
                testing=False,
            ),
            make_dataloader(
                val_df,
                track_idx2embeds,
                cfg,
                testing=False,
            ),
        )
