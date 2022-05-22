import torch
import torch.nn as nn
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)
    if sets_sum.item() == 0:
        return torch.tensor(1, dtype=torch.float32).to(input.device)
    else:
        return (2 * inter + epsilon) / (sets_sum + epsilon)


def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = torch.tensor(0, dtype=torch.float32).to(input.device)
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...], target[:, channel, ...].float(), epsilon
        )
    return dice / input.shape[1]


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        return 1 - multiclass_dice_coeff(input, target)
