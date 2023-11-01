import numpy as np


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
            size = max(self.max_len, len(embeds))
            indices = np.random.choice(
                len(embeds), size=size, replace=self.params.get("with_repeats", False)
            )
            return [embeds[i] for i in indices]
        return embeds


class AddNoise(Augmentation):
    def __call__(self, embeds):
        if self._aug_condition():
            p = self.params.get("p", 0.03)
            return [self._add_noise(e, p) for e in embeds]
        return embeds

    def _add_noise(self, embed, p):
        if np.random.random() < p:
            a_min = embed.min()
            a_max = embed.max()
            noise = max(self.params.get("noise", 0.01), 0)
            scale = (a_max - a_min) * noise
            embed += np.random.normal(loc=0, scale=scale, size=len(embed))
        return embed


class AddInversions(Augmentation):
    def __call__(self, embeds):
        if self._aug_condition():
            p = self.params.get("p", 0.03)
            probas = np.random.random(size=len(embeds) - 1)
            indices = np.where(probas < p)[0]
            for i in indices:
                embeds[i], embeds[i + 1] = embeds[i + 1], embeds[i]
            return embeds


class AugmentationList:
    NAMES = {
        "AddInversions": AddInversions,
        "AddNoise": AddNoise,
        "MixUp": MixUp,
    }

    def __init__(self, augmentations, max_len=None):
        self.augmentations = [
            self.NAMES[a](params, max_len) for a, params in augmentations.items()
        ]

    def __call__(self, embeds):
        for aug in self.augmentations:
            embeds = aug(embeds)
        return embeds
