import torch
from torch.utils.data import Dataset
import numpy as np
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
        if self.testing:
            return track_idx, embeds
        tags = [int(x) for x in row.tags.split(",")]
        target = np.zeros(NUM_TAGS)
        target[tags] = 1
        return track_idx, embeds, target


def collate_fn(b):
    track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
    embeds = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(x[1]) for x in b], batch_first=True
    )
    targets = np.vstack([x[2] for x in b])
    targets = torch.from_numpy(targets)
    return track_idxs, embeds, targets


def collate_fn_test(b):
    track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
    embeds = torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(x[1]) for x in b], batch_first=True
    )
    return track_idxs, embeds
