"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import F1_Loss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 1000000
TARGET_ANGLE = torch.as_tensor([np.radians(135)]).cuda()

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # class_weights = torch.ones(cfg.MODEL.FCOS.NUM_CLASSES).cuda()
        # class_weights[1] = 0.4
        # self.cls_loss_func_ce = nn.CrossEntropyLoss(reduction="sum")
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.f1_loss = F1_Loss(num_classes=cfg.MODEL.FCOS.NUM_CLASSES)

    def get_sample_region(self, gt, centers, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        # num_gts = gt.shape[0]
        # # print('Num gts: {}'.format(num_gts))
        # K = len(gt_xs)
        # # print('K {}'.format(K))
        # gt = gt[None].expand(K, num_gts, 8)
        # # print('gt {}'.format(gt.shape))
        # center_x = centers[:,0].expand(K, num_gts)
        # # print('center x {}'.format(center_x.shape))
        # center_y = centers[:,1].expand(K, num_gts)
        # center_gt = gt.new_zeros(gt.shape)
        # # print('center_gt {}'.format(center_gt.shape))
        # # no gt
        # if center_x[..., 0].sum() == 0:
        #     return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        # beg = 0
        # for level, n_p in enumerate(num_points_per):
        #     end = beg + n_p
        #     stride = strides[level] * radius
        #     xmin = center_x[beg:end] - stride
        #     ymin = center_y[beg:end] - stride
        #     xmax = center_x[beg:end] + stride
        #     ymax = center_y[beg:end] + stride
        #     # limit sample region in gt
        #     center_gt[beg:end, :, 0] = torch.where(
        #         xmax > gt[beg:end, :, 0], gt[beg:end, :, 0], xmax
        #     )
        #     center_gt[beg:end, :, 1] = torch.where(
        #         ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
        #     )
        #     # center_gt[beg:end, :, 2] = torch.where(
        #     #     xmax > gt[beg:end, :, 2],
        #     #     gt[beg:end, :, 2], xmax
        #     # )
        #     # center_gt[beg:end, :, 3] = torch.where(
        #     #     ymin > gt[beg:end, :, 3], ymin,
        #     #     gt[beg:end, :, 3]
        #     # )
        #     center_gt[beg:end, :, 2] = torch.where(
        #         xmax > gt[beg:end, :, 2], xmin,
        #         gt[beg:end, :, 2]
        #     )
        #     center_gt[beg:end, :, 3] = torch.where(
        #         ymax > gt[beg:end, :, 3], ymin,
        #         gt[beg:end, :, 3]
        #     )
        #     # center_gt[beg:end, :, 6] = torch.where(
        #     #     xmax > gt[beg:end, :, 6], xmin,
        #     #     gt[beg:end, :, 6]
        #     # )
        #     # center_gt[beg:end, :, 7] = torch.where(
        #     #     ymax > gt[beg:end, :, 7], ymin,
        #     #     gt[beg:end, :, 7]
        #     # )
        #     center_gt[beg:end, :, 4] = torch.where(
        #         xmax > gt[beg:end, :, 4], xmin,
        #         gt[beg:end, :, 4]
        #     )
        #     center_gt[beg:end, :, 5] = torch.where(
        #         ymax > gt[beg:end, :, 5],
        #         gt[beg:end, :, 5], ymax
        #     )
        #     # center_gt[beg:end, :, 10] = torch.where(
        #     #     xmax > gt[beg:end, :, 10], xmin,
        #     #     gt[beg:end, :, 10]
        #     # )
        #     # center_gt[beg:end, :, 11] = torch.where(
        #     #     ymax > gt[beg:end, :, 11],
        #     #     gt[beg:end, :, 11], ymax
        #     # )           
        #     center_gt[beg:end, :, 6] = torch.where(
        #         xmax > gt[beg:end, :, 6],
        #         gt[beg:end, :, 6], xmax
        #     )
        #     center_gt[beg:end, :, 7] = torch.where(
        #         ymax > gt[beg:end, :, 7],
        #         gt[beg:end, :, 7], ymax
        #     )
        #     # center_gt[beg:end, :, 14] = torch.where(
        #     #     xmax > gt[beg:end, :, 14],
        #     #     gt[beg:end, :, 14], xmax
        #     # )
        #     # center_gt[beg:end, :, 15] = torch.where(
        #     #     ymax > gt[beg:end, :, 15],
        #     #     gt[beg:end, :, 15], ymax
        #     # )
        #     beg = end
        # r_x = center_gt[..., 0] - gt_xs[:, None]
        # r_y = gt_ys[:, None] - center_gt[..., 1]
        # # tr_x = center_gt[..., 2] - gt_xs[:, None]
        # # tr_y = gt_ys[:, None] - center_gt[..., 3]
        # t_x = gt_xs[:, None] - center_gt[..., 2]
        # t_y = gt_ys[:, None] - center_gt[..., 3]
        # # tl_x = gt_xs[:, None] - center_gt[..., 6]
        # # tl_y = gt_ys[:, None] - center_gt[..., 7]
        # l_x = gt_xs[:, None] - center_gt[..., 4]
        # l_y = center_gt[..., 5] - gt_ys[:, None]
        # # bl_x = gt_xs[:, None] - center_gt[..., 10]
        # # bl_y = center_gt[..., 11] - gt_ys[:, None]
        # b_x = center_gt[..., 6] - gt_xs[:, None]
        # b_y = center_gt[..., 7] - gt_ys[:, None]
        # # br_x = center_gt[..., 14] - gt_xs[:, None]
        # # br_y = center_gt[..., 15] - gt_ys[:, None]

        # # center_bbox = torch.stack((r_x, r_y, tr_x, tr_y, t_x, t_y, tl_x, tl_y, l_x, l_y, bl_x, bl_y, b_x, b_y, br_x, br_y), -1)
        # center_bbox = torch.stack((r_x, r_y, t_x, t_y, l_x, l_y, b_x, b_y), -1)
        # inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        # # print(inside_gt_bbox_mask.shape)
        # return inside_gt_bbox_mask

        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = centers[:,0].expand(K, num_gts)
        center_y = centers[:,1].expand(K, num_gts)
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            # center_gt[beg:end, :, 4] = torch.where(
            #     xmax > gt[beg:end, :, 4], gt[beg:end, :, 4], xmax
            # )
            # center_gt[beg:end, :, 5] = torch.where(
            #     ymax > gt[beg:end, :, 5], gt[beg:end, :, 5], ymax
            # )
            # center_gt[beg:end, :, 6] = torch.where(
            #     xmin > gt[beg:end, :, 6], xmin,
            #     gt[beg:end, :, 6]
            # )
            # center_gt[beg:end, :, 7] = torch.where(
            #     ymin > gt[beg:end, :, 7], ymin,
            #     gt[beg:end, :, 7]
            # )
            beg = end
        left_x = gt_xs[:, None] - center_gt[..., 0]
        right_x = center_gt[..., 2] - gt_xs[:, None]
        left_y = gt_ys[:, None] - center_gt[..., 1]
        right_y = center_gt[..., 3] - gt_ys[:, None]
        # top_x = gt_xs[:, None] - center_gt[...,4]
        # bottom_x = center_gt[...,6] - gt_xs[:, None]
        # top_y = gt_ys[:, None] - center_gt[...,5]
        # bottom_y = center_gt[...,7] - gt_xs[:, None]
        center_bbox = torch.stack((left_x, right_x, left_y, right_y), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        # print(locations.size())
        # exit(1)
        xs, ys = locations[:, 0], locations[:, 1]

        # print('No of Boxes: {}'.format(len(targets)))
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            diags = targets_per_im.get_field("diagonals")
            labels_per_im = targets_per_im.get_field("labels")
            centers = targets_per_im.get_field("centers")
            area = targets_per_im.area()

            r_x = diags[:, 0][None] - xs[:, None]
            r_y = ys[:, None] - diags[:, 1][None]
            tr_x = diags[:, 2][None] - xs[:, None]
            tr_y = ys[:, None] - diags[:, 3][None]
            t_x = xs[:, None] - diags[:, 4][None]
            t_y = ys[:, None] - diags[:, 5][None]
            tl_x = xs[:, None] - diags[:, 6][None]
            tl_y = ys[:, None] - diags[:, 7][None]
            l_x = xs[:, None] - diags[:, 8][None]
            l_y = diags[:, 9][None] - ys[:, None]
            bl_x = xs[:, None] - diags[:, 10][None]
            bl_y = diags[:, 11][None] - ys[:, None]
            b_x = diags[:, 12][None] - xs[:, None]
            b_y = diags[:, 13][None] - ys[:, None]
            br_x = diags[:, 14][None] - xs[:, None]
            br_y = diags[:, 15][None] - ys[:, None]


            reg_targets_per_im = torch.stack([r_x, r_y, tr_x, tr_y, t_x, t_y, tl_x, tl_y, l_x, l_y, bl_x, bl_y, b_x, b_y, br_x, br_y], dim=2)

            # print('Reg Target: {}'.format(reg_targets_per_im.size()))
            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    diags[:,[8,9,0,1]],
                    centers,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            # else:
            # no center sampling, it will use all the locations within a ground-truth box
            # is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # torch.set_printoptions(threshold=500000)
            # print(is_in_boxes.shape)
            # exit(1)

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # print(locations_to_gt_area.size())

            # print('Loc to GT size: {}'.format(locations_to_gt_area.size()))
            try:
                locations_to_gt_area[is_in_boxes == 0] = INF
                # test_loc = (locations_to_gt_area < INF).nonzero()

            except IndexError:
                print('Is in boxes: {}'.format(is_in_boxes.size()))
                print('Reg targets: {}'.format(reg_targets_per_im.size()))
                print(len(area))
                exit(1)
            try:
                locations_to_gt_area[is_cared_in_the_level == 0] = INF
            except IndexError:
                print(reg_targets_per_im.size())
                print(is_cared_in_the_level.size())
                print(len(area))
                exit(1)

            # test_loc = (locations_to_gt_area < INF).nonzero()
            # print(test_loc.size())
            # exit(1)

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            # print('Reg Target Shape: {}'.format(reg_targets_per_im.size()))
        return labels, reg_targets

    def compute_convexity_targets(self, box_reg):
        box_reg_pred = box_reg.detach()
        box_reg_pred = box_reg_pred.view(-1,8,2)
        all_angles_pred = torch.zeros(len(box_reg), 8).float().cuda()
        for j in range(8):
            pred_p1 = box_reg_pred[:,j,:]
            pred_p2 = box_reg_pred[:,(j+1)%8,:]
            pred_p3 = box_reg_pred[:, (j+2)%8,:]

            pred_p2p1 = torch.norm(pred_p2-pred_p1, dim=1)
            pred_p2p3 = torch.norm(pred_p2-pred_p3, dim=1)
            pred_p1p3 = torch.norm(pred_p1-pred_p3, dim=1)

            #Cosine Rule to find angle
            all_angles_pred[:,j] = torch.acos(pred_p2p1**2 + pred_p2p3**2 - pred_p1p3**2)/(2 * pred_p2p1 * pred_p2p3)

        all_angles_diff = torch.abs(TARGET_ANGLE - all_angles_pred) + 1e-8
        # all_angles_diff_copy = all_angles_diff
        all_angles_diff[all_angles_diff != all_angles_diff] = 0
        mask = (all_angles_diff > 0).float()
        convexity_targets = (torch.sum(all_angles_diff, 1)/torch.max(mask.sum(1), torch.ones(mask.size()[0]).cuda()))*5
        return convexity_targets

    def __call__(self, locations, box_cls, box_regression, targets):  #centerness
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)


        box_cls_flatten = []
        box_regression_flatten = []
        # centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 16))
            labels_flatten.append(labels[l].reshape(-1))
            diags = reg_targets[l]
            reg_targets_flatten.append(diags.reshape(-1, 16)) #To verify it's an 16 long sized vector. Does not do anything.
            
            # centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        # centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        # centerness_flatten = centerness_flatten[pos_inds]

        is_nans = torch.isnan(reg_targets_flatten)
        if is_nans.sum() > 0:
            print(targets)
            print('for target')
            exit(1)

        # is_nans = torch.isnan(box_regression_flatten)
        # if is_nans.sum() > 0:
        #     print(targets)
        #     print('for prediction')
        #     exit(1)

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # torch.set_printoptions(threshold=50000)

        cls_loss = (self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        )/ num_pos_avg_per_gpu)# + self.f1_loss(box_cls_flatten, labels_flatten.long())

        # cls_loss = self.f1_loss(box_cls_flatten, labels_flatten.long())

        if pos_inds.numel() > 0:
            # centerness_targets = self.compute_convexity_targets(box_regression_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            # sum_centerness_targets_avg_per_gpu = \
            #     reduce_sum(centerness_targets.mean()).item() / float(num_gpus)

            # print(box_regression_flatten)
            # print(reg_targets_flatten)
            reg_loss_inter = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten
                # centerness_targets
            )
            # print(reg_loss_inter)
            reg_loss = reg_loss_inter#/ (sum_centerness_targets_avg_per_gpu)
            # print(reg_loss)
            # convexity_loss = self.convexity_loss_func(
            #     centerness_flatten,
            #     centerness_targets
            # ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            # reduce_sum(centerness_flatten.new_tensor([0.0]))
            # convexity_loss = centerness_flatten.sum()

        # print(cls_loss, reg_loss, convexity_loss)
        return cls_loss, reg_loss #, convexity_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
