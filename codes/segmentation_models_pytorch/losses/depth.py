import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class DepthLoss(_Loss):
    def __init__(self):
        super(DepthLoss, self).__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)
        
        bs = y_true.size(0)
        y_true = y_true.view(bs,-1)
        y_pred = y_pred.view(bs,-1)
        loss = torch.linalg.norm(y_true-y_pred,p=1,dim=1)/bs
        
        return loss