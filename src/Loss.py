import torch
from torch import nn
from torch.nn import BCELoss
import torch.nn.functional as F
import numpy as np
from hyptorch.pmath import dist_matrix


def contrastive_loss(x0, x1, tau, hyper_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    # dist = dist_f(x0, x0)
    # dist2 = -euclidean_distance(x0, x0)
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits00_numpy = logits00.detach().cpu().numpy()
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    stats = {
        "logits/min": logits01.min().item(),
        "logits/mean": logits01.mean().item(),
        "logits/max": logits01.max().item(),
        "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
        "logits": logits,
    }
    return loss, stats




def false_negative_mask(center_index_level):
    bs = center_index_level.shape[0]
    mask = torch.zeros(bs, bs, dtype=torch.uint8)
    for i in range(bs):
        for j in range(bs):
            mask[i][j] = 1 if center_index_level[i] == center_index_level[j] and i != j else 0
    return torch.cat([mask, mask], dim=1)



def hierarchical_contrastive_loss(x0, x1, tau, center_index, hyper_c):
    level = center_index.shape[1]
    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y)
    loss_all = 0
    bsize = x0.shape[0]
    for i in range(level):
        # define false negative mask
        center_index_level = center_index[:, i]
        mask = false_negative_mask(center_index_level).cuda()
        mask_numpy = mask.detach().cpu().numpy()
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        # dist = dist_f(x0, x0)
        # dist2 = -euclidean_distance(x0, x0)
        logits00 = dist_f(x0, x0) / tau - eye_mask
        logits01 = dist_f(x0, x1) / tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits_numpy = logits.detach().cpu().numpy()
        logits = logits - mask * 1e9
        logits_numpy = logits.detach().cpu().numpy()
        logits_numpy = logits.detach().cpu().numpy()
        positive = torch.diag(logits01)
        loss = torch.mean(-torch.log(torch.exp(positive) / torch.sum(torch.exp(logits), dim=1)))

        loss_all += (1 / (i + 1)) * loss
    return loss_all / level



def parse_proto(results, num_cluster):
    """
    center_corresponding: NxLx128
    center_index : NxL center assignment of each sample at different hierarchies.
    :param results:
    :param num_cluster:
    :return:
    """
    level = len(num_cluster)
    dim = results['centroids'][0].shape[1]
    num_samples = results['im2cluster'][0].shape[0]
    center_index = np.zeros(shape=(num_samples, level))
    center_corresponding = np.zeros(shape=(num_samples, level, dim))
    for i in range(level):
        if i == 0:
            center_index[:, 0] = results['im2cluster'][0]
        else:
            center_index[:, i] = results['im2cluster'][i][center_index[:, i - 1].astype('int64')]
    for j in range(num_samples):
        for k in range(level):
            center_corresponding[j, k, :] = results['centroids'][k][center_index[j, k].astype('int64')]
    return center_corresponding, center_index


def contrastive_proto(z_i, z_j, center_corresponding, center_index, results, tau=0.2, hyper_c=0):
    centers = results['centroids']
    level = len(centers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    center_corresponding = torch.tensor(center_corresponding, dtype=torch.float32).to(device)
    # centers = torch.tensor(centers).to(device)
    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, hyper_c)
    loss_all = 0
    for l in range(level):
        centers_level = torch.tensor(centers[l], dtype=torch.float32).to(device)
        positive_i = torch.diag(torch.exp(dist_f(z_i, center_corresponding[:, l, :]) / tau))
        negative_i = torch.sum(torch.exp(dist_f(z_i, centers_level) / tau), dim=1) + torch.tensor(1e-5).to(device)

        positive_j = torch.diag(torch.exp(dist_f(z_j, center_corresponding[:, l, :]) / tau))
        negative_j = torch.sum(torch.exp(dist_f(z_j, centers_level) / tau), dim=1) + torch.tensor(1e-5).to(device)

        loss_level = -torch.log(positive_i / negative_i) - torch.log(positive_j / negative_j)
        loss_all += (1 / (l + 1)) * loss_level.mean()
    return loss_all / level


if __name__ == '__main__':
    a = torch.tensor([[0, 1], [1, 1], [0, 1], [1, 0]])
    # b=torch.eye(1024)
    print()
