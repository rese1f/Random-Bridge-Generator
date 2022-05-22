import torch
import torch.nn as nn
from torch import Tensor

class depth_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, input: Tensor, target: Tensor):
        
        # depth loss for regression
        input = input.squeeze()
        assert input.size() == target.size()
        # loss = torch.norm(torch.log(input/target))
        # import pdb; pdb.set_trace()
        loss = self.loss(input, target)
        return loss