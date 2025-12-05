import torch

checkpoint = torch.load("checkpoints_highacc_run/best_model.pth")

print(type(checkpoint))
print(checkpoint.keys())