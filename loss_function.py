import torch
import torch.nn as nn


celoss = nn.CrossEntropyLoss(reduction='mean')
mseloss = nn.MSELoss(reduction='mean')


def topk_loss(instances_logits,  bag_labels, k, device, loss_type='CE'):

    logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)

    # as drop_last == False, we need to consider the size of the last batch
    real_size = int(instances_logits.shape[0])

    for i in range(real_size):
        tmp, tmp_index = torch.topk(instances_logits[i], k=3, dim=0)
        logits = torch.cat((logits, tmp), dim=0)

        if bag_labels[i] == 1:
            real_label = torch.cat([real_label, torch.ones(k, 1).to(device)], dim=0)
        else:
            real_label = torch.cat([real_label, torch.zeros((k, 1)).to(device)], dim=0)

    if loss_type == 'CE':
        milloss = celoss(input=logits, target=real_label)
        return milloss
    elif loss_type == 'MSE':
        milloss = mseloss(input=logits, target=real_label)
        return milloss


def center_loss(instances_logits, bag_labels, device):

    normal_smooth_loss = torch.zeros(0).to(device)

    real_size = int(instances_logits.shape[0])

    for i in range(real_size):
        if bag_labels[i] == 0:
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(instances_logits[i]).unsqueeze(0)))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)
    return normal_smooth_loss
