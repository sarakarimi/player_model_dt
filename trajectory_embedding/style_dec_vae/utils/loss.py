import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import argmax


class MiniGridLoss(_Loss):
    def __init__(self):
        super(MiniGridLoss, self).__init__()

    def forward(self, input, target):
        """
         Custom loss function for different class regions in mingrid with onehot wrapper
         The one hot observation wrapper converts observations into 3 one-hot encoded parts and concatenates them to each other!
         """
        # input = input.view(-1, 25, 20)
        # target = target.view(-1, 25, 20)
        criterion = nn.MSELoss(reduction='none')

        # class_1_loss = F.nll_loss(
        #     F.log_softmax(input[:, :, 0:11] + 1e-8, dim=2).view(-1, 11),
        #     argmax(target[:, :, 0:11], dim=2).view(-1),
        #     # reduction='sum'
        # )
        # part_1_loss = criterion(input[:, :, 0:11], target[:, :, 0:11])

        # class_2_loss = F.nll_loss(
        #     F.log_softmax(input[:, :, 11:17] + 1e-8, dim=2).view(-1, 6),
        #     argmax(target[:, :, 11:17], dim=2).view(-1),
        #     # reduction='sum'
        # )
        # part_2_loss = criterion(input[:, :, 11:17], target[:, :, 11:17])

        # class_3_loss = F.nll_loss(
        #     F.log_softmax(input[:, :, 17:20] + 1e-8, dim=2).view(-1, 3),
        #     argmax(target[:, :, 17:20], dim=2).view(-1),
        #     # reduction='sum'
        # )
        # part_3_loss = criterion(input[:, :, 17:20], target[:, :, 17:20])
        # loss = torch.cat([part_1_loss, part_2_loss, part_3_loss], dim=-1)
        loss = criterion(input, target)
        loss = loss.sum() #* 500
        # loss /= 500
        return loss
