import numpy as np
import torch
from skimage.transform import resize

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    # 计算高斯核的中心坐标
    m, n = [(ss - 1.) / 2. for ss in shape]
    # 生成网格坐标
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # 计算二维高斯分布
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # 将小于某个阈值的值设为0
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, bh, bw, k=1):
    # 设置高斯核的宽和高
    bh = radius
    bw = radius
    radius = radius
    diameter = 2 * radius + 1
    # 生成二维高斯分布
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # gaussian = resize(gaussian, (bh*2+1, bw*2+1))
    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    # 计算在热图中高斯分布的范围
    left, right = min(x, bw), min(width - x, bw + 1)
    top, bottom = min(y, bh), min(height - y, bh + 1)
    # 获取热图中需要覆盖的区域
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[bh - top:bh + bottom, bw - left:bw + right]
    # 将高斯分布覆盖到热图中
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def heamp_focal_loss(pred, target):
    import pdb
    pdb.set_trace()
    # 为了变成[B, H, W, C]格式
    # pred = pred.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)
    pred = pred[:, :, :, 0].unsqueeze(3)
    device = torch.device('cuda')
    pred = pred.to(device)

    # -------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    # -------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()  # eq为等于
    pos_inds = pos_inds.to(device)
    neg_inds = target.lt(1).float()  # lt为小于
    neg_inds = neg_inds.to(device)
    # -------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    # -------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)  # 求指数

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)  # 将数字压缩到[1e-6, 1-1e-6]区间内
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    num_neg = neg_inds.float().sum()
    pos_loss = pos_loss.sum()
    pos_loss = pos_loss * ((num_pos + num_neg) / num_pos)
    neg_loss = neg_loss.sum()
    neg_loss = neg_loss * ((num_pos + num_neg) / num_neg)

    if num_pos == 0:
        loss = -neg_loss / num_neg
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss
    #
    # # -------------------------------------------------------------------------#
    # #   进行损失的归一化
    # # -------------------------------------------------------------------------#
    # num_pos = pos_inds.float().sum()
    # pos_loss = pos_loss.sum()
    # neg_loss = neg_loss.sum()
    #
    # if num_pos == 0:
    #     loss = -neg_loss
    # else:
    #     loss = -(pos_loss + neg_loss) / num_pos
    # return loss