# GIoU and Linear IoU are added by following
# https://github.com/yqyao/FCOS_PLUS/blob/master/maskrcnn_benchmark/layers/iou_loss.py.
import torch
from torch import nn
import numpy as np
from shapely.geometry import Polygon
from shapely.geos import TopologicalError


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        assert len(pred) == len(target), "Dimension mismatch for IOU Loss computation"

        loss = torch.sum(torch.pow((pred - target), 2), 1)/pred.size()[1]
        loss = torch.sqrt(loss)

        # if weight is not None and weight.sum() > 0:
        #     loss = loss * (weight + 1e-8)
        loss = torch.mean(loss)
        return loss
        # try:
        #     pred_numpy = pred.cpu().data.numpy()
        #     target_numpy = target.cpu().data.numpy()

        #     pred_numpy = pred_numpy.reshape((-1,8,2))
        #     target_numpy = target_numpy.reshape((-1,8,2))

        #     intersection_area = np.zeros(pred_numpy.shape[0])

        #     for instance in range(pred_numpy.shape[0]):
        #         intersection_area[instance] = Polygon(pred_numpy[instance]).intersection(Polygon(target_numpy[instance])).area

        #     intersection_area = torch.as_tensor(intersection_area, dtype=torch.float).cuda()

        # except TopologicalError:
        #     pred_pseudo = pred.reshape((-1,8,2))
        #     target_pseudo = target.reshape((-1,8,2))
        # instances = pred.size()[0]

        # x_max_pred, _ = torch.max(pred[:, ::2], 1)
        # x_min_pred, _ = torch.min(pred[:, ::2], 1)
        # y_max_pred, _ = torch.max(pred[:, 1::2], 1)
        # y_min_pred, _ = torch.min(pred[:, 1::2], 1)

        # x_max_target, _ = torch.max(target[:, ::2], 1)
        # x_min_target, _ = torch.min(target[:, ::2], 1)
        # y_max_target, _ = torch.max(target[:, 1::2], 1)
        # y_min_target, _ = torch.min(target[:, 1::2], 1)

        # x_max = torch.min(x_max_pred, x_max_target)
        # x_min = torch.max(x_min_pred, x_min_target)
        # y_max = torch.min(y_max_pred, y_max_target)
        # y_min = torch.max(y_min_pred, y_max_target)

        # intersection_area = torch.max(torch.zeros(instances).cuda(), (x_max - x_min + 1)) * torch.max(torch.zeros(instances).cuda(), (y_max - y_min + 1))

        # pred_area = (x_max_pred - x_min_pred + 1) * (y_max_pred - y_min_pred + 1)
        # target_area = (x_max_target - x_min_target + 1) * (y_max_target - y_min_target + 1)

        # vertices = pred.size()[1]
        # pred_area = torch.zeros(pred.size()[0]).cuda()
        # target_area = torch.zeros(target.size()[0]).cuda()

        # #Cycle through the x-y pairs for each instance to compute the areas mod((x1y2 - y1x2) + ... + (xny1 - ynx1)/2)
        # for vertex in range(8):
        #     pred_area += pred[:, vertex*2]*pred[:, ((vertex+1)*2+1)%(vertices)] - pred[:, vertex*2+1]*pred[:, ((vertex+1)*2)%(vertices)]
        #     target_area += target[:, vertex*2]*target[:, ((vertex+1)*2+1)%(vertices)] - target[:, vertex*2+1]*target[:, ((vertex+1)*2)%(vertices)]

        # pred_area = torch.abs(pred_area)/2
        # target_area = torch.abs(target_area)/2

        # area_union = target_area + pred_area - intersection_area

        # ious = (intersection_area + 1.0)/(area_union + 1.0)


        # is_nans = torch.isnan(ious).sum()

        # if is_nans:
        #     print(pred)
        #     print(target)
        #     print(pred_target_intersects)

        # if self.loss_type == 'iou':
        #     losses = -torch.log(ious)
        # elif self.loss_type == 'linear_iou':
        #     losses = 1 - ious
        # elif self.loss_type == 'giou':
        #     # losses = 1 - gious
        #     raise Exception('No GIoU for polygon vertices implemented yet')
        # else:
        #     raise NotImplementedError

        # if weight is not None and weight.sum() > 0:
        #     return (losses * weight).sum()
        # else:
        #     assert losses.numel() != 0
        #     return losses.sum()
                