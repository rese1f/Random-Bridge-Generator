import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
    
    """how to use it?

    model = Model()

    awl = AutomaticWeightedLoss(2)	# we have 2 losses
    loss_1 = ...
    loss_2 = ...

    # learnable parameters
    optimizer = optim.Adam([
                    {'params': model.parameters()},
                    {'params': awl.parameters(), 'weight_decay': 0}
                ])

    for i in range(epoch):
        for data, label1, label2 in data_loader:
            # forward
            pred1, pred2 = Model(data)	
            # calculate losses
            loss1 = loss_1(pred1, label1)
            loss2 = loss_2(pred2, label2)
            # weigh losses
            loss_sum = awl(loss1, loss2)
            # backward
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
    """