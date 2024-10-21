import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import argmax


class MiniGridLoss(_Loss):
    def __init__(self):
        super(MiniGridLoss, self).__init__()

    def forward(self, input, target):
        """ Custom loss function with NLL for different class regions in mingrid with onehot wrapper """
        input = input.view(-1, 25, 20)
        target = target.view(-1, 25, 20)

        class_1_loss = F.nll_loss(
            F.log_softmax(input[:, :, 0:11] + 1e-8, dim=2).view(-1, 11),
            argmax(target[:, :, 0:11], dim=2).view(-1)
        )
        # print(class_1_loss)

        class_2_loss = F.nll_loss(
            F.log_softmax(input[:, :, 11:17] + 1e-8, dim=2).view(-1, 6),
            argmax(target[:, :, 11:17], dim=2).view(-1)
        )
        # print(class_2_loss)

        class_3_loss = F.nll_loss(
            F.log_softmax(input[:, :, 17:20] + 1e-8, dim=2).view(-1, 3),
            argmax(target[:, :, 17:20], dim=2).view(-1)
        )
        # print(class_3_loss)
        return class_1_loss + class_2_loss + class_3_loss
