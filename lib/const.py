import torch

NUM_TAGS = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
