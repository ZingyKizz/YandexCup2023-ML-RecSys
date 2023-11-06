import numpy as np
import torch


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        item = iterable[ndx : ndx + n]
        if len(item) == n:
            yield item


class Augmentation:
    def __init__(self, params, max_len):
        self.params = params or {}
        self.max_len = max_len or 1024

    def _aug_condition(self):
        p = self.params.get("proba", 0)
        if p == 0:
            return False
        return np.random.random() < p


class MixUp(Augmentation):
    def __call__(self, embeds):
        if self._aug_condition():
            return np.random.permutation(embeds)
        return embeds


class AddNoise(Augmentation):
    def __call__(self, embeds):
        if self._aug_condition():
            embeds = embeds.copy()
            p = self.params.get("p", 0.03)
            indices = np.where(np.random.random(size=len(embeds)) < p)
            embeds[indices] += np.random.normal(
                loc=0,
                scale=self.params.get("noise", 0.01),
                size=(len(indices), embeds.shape[-1]),
            )
        return embeds


class AddInversions(Augmentation):
    def __call__(self, embeds):
        if self._aug_condition():
            embeds = embeds.copy()
            p = self.params.get("p", 0.03)
            probas = np.random.random(size=len(embeds) - 1)
            indices = np.where(probas < p)[0]
            for i in indices:
                embeds[i], embeds[i + 1] = embeds[i + 1], embeds[i]
        return embeds


class TrueMixUp(Augmentation):
    def __call__(self, x, y=None):
        if y is None:
            return x
        alpha = self.params.get("alpha", 0.5)
        mixups_per_batch = self.params.get("mixups_per_batch", 1)
        if self._aug_condition():
            x, y = torch.clone(x), torch.clone(y)
            batch_size = x.size(0)
            mixup_indices_length = min(batch_size, mixups_per_batch * 2)
            mixup_indices = torch.randperm(batch_size)[:mixup_indices_length]
            for i, j in batch(mixup_indices, 2):
                x[i] = alpha * x[i] + (1 - alpha) * x[j]
                y[i] = alpha * y[i] + (1 - alpha) * y[j]
        return x, y


class AugmentationList:
    NAMES = {
        "AddInversions": AddInversions,
        "AddNoise": AddNoise,
        "MixUp": MixUp,
        "TrueMixUp": TrueMixUp,
    }

    def __init__(self, augmentations, max_len=None):
        self.augmentations = {
            a: self.NAMES[a](params, max_len) for a, params in augmentations.items()
        }

    def __call__(self, embeds):
        for aug in self.augmentations.values():
            embeds = aug(embeds)
        return embeds

    def __getitem__(self, name):
        return self.augmentations[name]

    def __contains__(self, name):
        return name in self.augmentations
