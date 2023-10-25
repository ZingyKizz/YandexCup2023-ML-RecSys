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


class Collator:
    def __init__(self, max_len=None, testing=False):
        self.testing = testing
        self.max_len = max_len

    def __call__(self, b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
        embeds = [torch.from_numpy(x[1][: self.max_len]) for x in b]
        padding_mask = self._create_padding_mask(embeds)
        embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
        if self.testing:
            return track_idxs, (embeds, padding_mask)
        targets = np.vstack([x[2] for x in b])
        targets = torch.from_numpy(targets)
        return track_idxs, (embeds, padding_mask), targets

    @staticmethod
    def _create_padding_mask(embeds):
        lens = torch.tensor([len(e) for e in embeds])
        padding_mask = torch.arange(max(lens))[None, :] >= lens[:, None]
        return padding_mask
