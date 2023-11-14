from torch.optim import Adam, AdamW
from lion_pytorch import Lion
from pytorch_optimizer import Tiger


def get_grouped_parameters(model, lr, alpha):
    params = list(model.named_parameters())

    def is_backbone(n):
        return "encoder" in n

    grouped_parameters = [
        {"params": [p for n, p in params if is_backbone(n)], "lr": lr},
        {"params": [p for n, p in params if not is_backbone(n)], "lr": alpha * lr},
    ]

    return grouped_parameters
