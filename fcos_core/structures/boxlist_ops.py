# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

from shapely.geometry import Polygon, MultiPolygon
from shapely.geos import TopologicalError
from .bounding_box import BoxList

from fcos_core.layers import nms as _box_nms
from tqdm import tqdm


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)

def diagonal_nms(boxlist, nms_thresh, nms_topk, score_field="scores"):
    boxes = boxlist.bbox
    mode = boxlist.mode
    img_size = boxlist.size
    diagonals = boxlist.get_field("diagonals")
    scores = boxlist.get_field(score_field)
    sorted_scores, sorted_ind = scores.sort(descending=True)
    sorted_diagonals = diagonals[sorted_ind]
    sorted_boxes = boxes[sorted_ind]
    kept_scores = []
    kept_diagonals = []
    kept_boxes = []


    sorted_scores = sorted_scores[:nms_topk].cpu().data.numpy()
    sorted_diagonals = sorted_diagonals[:nms_topk].cpu().data.numpy()
    sorted_boxes = sorted_boxes[:nms_topk].cpu().data.numpy()

    while len(sorted_scores) >= 1:
        rem_scores = []
        rem_diagonals = []
        rem_boxes = []

        curr_score = sorted_scores[0]
        curr_diagonal = sorted_diagonals[0]
        curr_box = sorted_boxes[0]

        kept_scores.append(curr_score)
        kept_diagonals.append(curr_diagonal)
        kept_boxes.append(curr_box)

        sorted_scores = sorted_scores[1:]
        sorted_diagonals = sorted_diagonals[1:]
        sorted_boxes = sorted_boxes[1:]
        
        for j in range(len(sorted_scores)):
            iou = calculate_diag_iou(curr_diagonal, sorted_diagonals[j])
            if iou < nms_thresh:
                rem_scores.append(sorted_scores[j])
                rem_diagonals.append(sorted_diagonals[j])
                rem_boxes.append(sorted_boxes[j])

        sorted_scores = rem_scores
        sorted_diagonals = rem_diagonals
        sorted_boxes = rem_boxes

    kept_scores = torch.as_tensor(kept_scores, dtype=torch.float)
    kept_diagonals = torch.as_tensor(kept_diagonals, dtype=torch.float)
    kept_boxes = torch.as_tensor(kept_boxes, dtype=torch.float)

    target = BoxList(kept_boxes, img_size, mode=mode)
    target.add_field("diagonals", kept_diagonals)
    target.add_field("scores", kept_scores)
    return target

def calculate_diag_iou(primary, secondary, eval=False):
    primary = primary.reshape((8,2))
    secondary = secondary.reshape((8,2))
    try:
        primary_poly = Polygon(primary)
        secondary_poly = Polygon(secondary)

        polygon_union_area = primary_poly.union(secondary_poly).area
        polygon_intersection_area = primary_poly.intersection(secondary_poly).area
    
    # angles = np.arange(10, 180, 45)
    # angles = np.radians(angles)

    # points_source = np.zeros((8,2))
    # points_candidate = np.zeros((8,2))
    # centroid_source = source[-2:]
    # centroid_candidate = candidate[-2:]

    # for j in range(angles.shape[0]):
    #     y_dist_pos_s = source[j]*np.sin(angles[j])
    #     y_dist_neg_s = source[j+4]*np.sin(angles[j])
    #     x_dist_pos_s = source[j]*np.cos(angles[j])
    #     x_dist_neg_s = source[j+4]*np.cos(angles[j])

    #     y_dist_pos_c = candidate[j]*np.sin(angles[j])
    #     y_dist_neg_c = candidate[j+4]*np.sin(angles[j])
    #     x_dist_pos_c = candidate[j]*np.cos(angles[j])
    #     x_dist_neg_c = candidate[j+4]*np.cos(angles[j])

    #     if j < 4:
    #         points_source[j,:] = x_dist_pos_s + centroid_source[0], y_dist_pos_s + centroid_source[1]
    #         points_source[j+4, :] = centroid_source[0] - x_dist_neg_s, centroid_source[1] - y_dist_neg_s
    #         points_candidate[j,:] = x_dist_pos_c + centroid_candidate[0], y_dist_pos_c + centroid_candidate[1]
    #         points_candidate[j+4, :] = centroid_candidate[0] - x_dist_neg_c, centroid_candidate[1] - y_dist_neg_c
    #     else:
    #         points_source[j, :] = centroid_source[0] - x_dist_pos_s, centroid_source[1] + y_dist_pos_s
    #         points_source[j + 4, :] = centroid_source[0] + x_dist_neg_s, centroid_source[1] - y_dist_neg_s
    #         points_candidate[j, :] = centroid_candidate[0] - x_dist_pos_c, centroid_candidate[1] + y_dist_pos_c
    #         points_candidate[j + 4, :] = centroid_candidate[0] + x_dist_neg_c, centroid_candidate[1] - y_dist_neg_c

    

    
    except TopologicalError:
        if not eval:
            polygon_union_area = 1.0
            polygon_intersection_area = 1.0
        else:
            polygon_union_area = 0.0
            polygon_intersection_area = 0.0
        
    # polygon_union_area = np.asarray(polygon_union_area)
    # polygon_intersection_area = np.asarray(polygon_intersection_area)
    iou = polygon_intersection_area/polygon_union_area
    # ious = torch.as_tensor(ious, dtype=torch.float).cuda()

    return iou

def all_class_nms(boxlist, nms_thresh):
    boxes = boxlist.bbox
    mode = boxlist.mode
    img_size = boxlist.size
    scores = boxlist.get_field("scores")
    labels = boxlist.get_field("labels")
    diagonals = boxlist.get_field("diagonals")

    sorted_scores, sorted_ind = scores.sort(descending=True)
    sorted_diagonals = diagonals[sorted_ind]
    sorted_boxes = boxes[sorted_ind]
    sorted_labels = labels[sorted_ind]

    sorted_scores = sorted_scores.cpu().data.numpy()
    sorted_diagonals = sorted_diagonals.cpu().data.numpy()
    sorted_boxes = sorted_boxes.cpu().data.numpy()
    sorted_labels = sorted_labels.cpu().data.numpy()

    kept_scores = []
    kept_boxes = []
    kept_labels = []
    kept_diagonals = []

    while len(sorted_scores) >= 1:
        rem_scores = []
        rem_diagonals = []
        rem_boxes = []
        rem_labels = []

        curr_score = sorted_scores[0]
        curr_diagonal = sorted_diagonals[0]
        curr_box = sorted_boxes[0]
        curr_label = sorted_labels[0]

        kept_scores.append(curr_score)
        kept_diagonals.append(curr_diagonal)
        kept_boxes.append(curr_box)
        kept_labels.append(curr_label)

        sorted_scores = sorted_scores[1:]
        sorted_diagonals = sorted_diagonals[1:]
        sorted_boxes = sorted_boxes[1:]
        sorted_labels = sorted_labels[1:]
        
        for j in range(len(sorted_scores)):
            iou = calculate_diag_iou(curr_diagonal, sorted_diagonals[j])
            if iou < nms_thresh:
                rem_scores.append(sorted_scores[j])
                rem_diagonals.append(sorted_diagonals[j])
                rem_boxes.append(sorted_boxes[j])
                rem_labels.append(sorted_labels[j])

        sorted_scores = rem_scores
        sorted_diagonals = rem_diagonals
        sorted_boxes = rem_boxes
        sorted_labels = rem_labels

    kept_scores = torch.as_tensor(kept_scores, dtype=torch.float)
    kept_diagonals = torch.as_tensor(kept_diagonals, dtype=torch.float)
    kept_boxes = torch.as_tensor(kept_boxes, dtype=torch.float)
    kept_labels = torch.as_tensor(kept_labels, dtype=torch.int64)

    target = BoxList(kept_boxes, img_size, mode=mode)
    target.add_field("diagonals", kept_diagonals)
    target.add_field("scores", kept_scores)
    target.add_field("labels", kept_labels)
    return target

def get_vertices_from_diags(boxlist):
    vertices = []
    diags = boxlist.get_field("diagonals")
    diags = diags.cpu().data.numpy()
    for i in range(len(diags)):
        vertices.append(diags[i].reshape((-1,2)))

    assert len(vertices) == diags.shape[0], 'Error, number of polygon instances dont match the number of diagonal instances while converting to vertices. Check your diagonal predictions'
    return np.asarray(vertices)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]

    try:
        return torch.cat(tensors, dim)
    except RuntimeError:
        print('OOPS again')
        # for tensor in tensors:
        #     print(tensor)
        exit(1)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)


    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
