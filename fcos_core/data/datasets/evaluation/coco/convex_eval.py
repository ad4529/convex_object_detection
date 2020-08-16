import torch
import numpy as np
from shapely.geometry import Polygon
from fcos_core.config import cfg
from .....structures.boxlist_ops import get_vertices_from_diags, calculate_diag_iou


def polygon_eval(dataset, predictions):

	diag_dict = dataset.diag_dict
	eval_thresh = cfg.MODEL.FCOS.EVAL_IOU_TH
	num_gts = 0
	all_preds = np.empty((0,16), dtype=np.float)
	all_scores = []
	tp = []

	for image_id, prediction in predictions.items():
		original_id = dataset.id_to_img_map[image_id]
		gt_diags = diag_dict[()][original_id][0]
		num_gts += len(gt_diags)

		scores = prediction.get_field('scores')
		prediction = prediction.get_field('diagonals')
		scores, sorted_ind = scores.sort(descending=True)
		preds = prediction[sorted_ind]

		scores = scores.cpu().data.numpy().tolist()
		preds = preds.cpu().data.numpy()

		temp_tp = np.zeros(len(scores))

		for gt_diag in gt_diags:
			for idx, pred in enumerate(preds):
				if temp_tp[idx] == 0:
					iou = calculate_diag_iou(gt_diag, pred, eval=True)
					if iou >= eval_thresh:
						temp_tp[idx] = 1
						break

		temp_tp = temp_tp.tolist()
		tp += temp_tp
		all_preds = np.append(all_preds, preds, axis=0)
		all_scores += scores


	return compute_apr(num_gts, all_scores, tp)


def compute_apr(num_gts, scores, tp):
	precision = []
	recall = []
	sorted_scores = sorted(scores, reverse=True)
	sorted_preds = [pred for _,pred in sorted(zip(scores, tp))]

	print(sum(tp))
	print(num_gts)

	gts_found = 0

	for idx,instance in enumerate(sorted_preds):
		if instance == 1:
			gts_found += 1
		precision.append(gts_found/(idx+1))
		recall.append(gts_found/num_gts)

	avg_precision = sum(precision)/len(precision)
	avg_recall = sum(recall)/len(recall)

	print('Average Precision: {}'.format(avg_precision))
	print('Average Recall: {}'.format(avg_recall))
	exit(1)