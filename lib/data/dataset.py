import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import KFold
from lib.const import NUM_TAGS
from lib.utils import make_instance
from lib.data.augmentations import AugmentationList


class TaggingDataset(Dataset):
    def __init__(self, df, track_idx2embeds, testing=False, weight_power=0.5):
        self.df = df
        self.track_idx2embeds = track_idx2embeds
        self.testing = testing
        self.weights = self._get_track_weights(df, weight_power)

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

    def _get_track_weights(self, df, weight_power):
        if self.testing:
            return None
        w = df.copy()
        w["tags"] = w["tags"].str.split(",")
        w = w.explode("tags")
        w["tag_cnt"] = w.groupby("tags").transform("count") / len(w)
        w["tag_weight"] = 1 / np.power(w["tag_cnt"], weight_power)
        res = df.merge(
            w.groupby("track", as_index=False)["tag_weight"].mean(), on=["track"]
        )
        weights = torch.from_numpy(res["tag_weight"].to_numpy())
        return weights


class Collator:
    def __init__(self, max_len=None, **kwargs):
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


class RandomMomentCollator:
    def __init__(self, max_len=None, testing=False, **kwargs):
        self.max_len = max_len
        self.testing = testing

    def __call__(self, b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
        embeds = [self._choose_random_moment_embedding(x[1]) for x in b]
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

    def _choose_random_moment_embedding(self, embedding):
        start = 0
        if not self.testing:
            start = np.random.randint(0, max(len(embedding) - self.max_len, 1))
        res = embedding[start : start + self.max_len]
        return torch.as_tensor(res)


def make_dataloader(
    df, track_idx2embeds, cfg, testing_dataset=False, testing_collator=False
):
    dataset = make_instance(
        cfg["dataset"],
        df=df,
        track_idx2embeds=track_idx2embeds,
        testing=testing_dataset,
        weight_power=cfg.get("dataset_weight_power", 0.5)
    )
    sampler = None
    if cfg.get("dataset_sample_weights", False) and (not testing_collator):
        sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=not testing_dataset,
        collate_fn=make_instance(
            cfg["collator"],
            max_len=cfg["max_len"],
            testing=testing_collator,
            augmentations=cfg.get("augmentations", {}),
        ),
        drop_last=False,
        sampler=sampler,
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
                testing_dataset=False,
                testing_collator=False,
            ),
            make_dataloader(
                val_df,
                track_idx2embeds,
                cfg,
                testing_dataset=False,
                testing_collator=True,
            ),
        )


class CollatorWithAug:
    def __init__(self, max_len=None, augmentations=None, testing=False, **kwargs):
        self.max_len = max_len
        self.augmentations = (
            AugmentationList(augmentations, max_len) if not testing else lambda x: x
        )
        self.testing = testing

    def __call__(self, b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
        embeds = [self.augmentations(x[1]) for x in b]
        embeds = [torch.from_numpy(e[: self.max_len]) for e in embeds]
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
