import torch.nn.functional as F
import torch
import logging


__all__ = ['softmax_weighted_loss', 'dice_loss', 'expand_target']

cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=6, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:] * target[:,i,:,:])
        l = torch.sum(output[:,i,:,:])
        r = torch.sum(target[:,i,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def softmax_weighted_loss(output, target, num_cls=6, eps=1e-7):
    target = target.float()
    B, _, H, W = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :]
        targeti = target[:, i, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2)) * 1.0 / (torch.sum(target, (1,2,3)) + eps))
        weighted = torch.reshape(weighted, (-1,1,1)).repeat(1,H,W)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss


def expand_target(x, n_class=6):
    """
        Converts NxHxW label image to NxCxHxW, where each label is stored in a separate channel
        :param input: 3D input image (NxHxW)
        :param C: number of channels/labels
        :return: 4D output image (NxCxHxW)
        """
    assert x.dim() == 3
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)

    for i in range(n_class):
      xx[:,i,:,:] = (x == i)

    return xx.to(x.device)
