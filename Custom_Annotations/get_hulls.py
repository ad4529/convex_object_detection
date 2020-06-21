""" Author and Maintainer : Abhisek Dey
    Master's Thesis Research
    This code converts MSCOCO segmentation annotations to it's corresponding convex hulls
    NOT FOR COMMERCIAL USE"""

import os
from pycocotools.coco import COCO
from pycocotools import mask
import matplotlib.pyplot as plt
import numpy as np
from imantics import Polygons, Mask
from scipy.spatial import ConvexHull
import argparse
import cv2
from tqdm import tqdm
import warnings


def create_hulls(coco, ids):
    ann_dict = {}
    tot_valid = 0
    tot_invalid = 0
    for idx in tqdm(ids):
        hulls = []
        invalids = []
        annId = coco.getAnnIds(imgIds=idx)
        anns = coco.loadAnns(annId)
        for pos,ann in enumerate(anns):
            # Check if there are segmentation masks for the instance
            if ann['segmentation']:
                # Check if the annotation is accurate
                if not ann['iscrowd']:
                    mask = coco.annToMask(ann)
                    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Adress the non-contiguous masks and if mask is valid
                    if len(contour):
                        if len(contour) == 1:
                            contour = contour[0]
                        else:
                            contour = np.concatenate(contour, axis=0)

                        contour = cv2.convexHull(contour)
                        contour = np.squeeze(contour)
                        # Check if the convex hull has at least 8 vertices
                        if contour.shape[0] >= 8:
                            interval = int(contour.shape[0]/8)
                            indices = np.arange(0,contour.shape[0],interval)
                            indices = indices[:8]
                            new_hull = contour[indices]
                            hulls.append(new_hull)
                            tot_valid += 1
                        else:
                            invalids.append(pos)
                            tot_invalid += 1
                    else:
                        invalids.append(pos)
                        tot_invalid += 1
                else:
                    invalids.append(pos)
                    tot_invalid += 1
            else:
                invalids.append(pos)
                tot_invalid += 1

        idx_ann = [hulls, invalids]
        ann_dict[idx] = idx_ann

    print('Total valid instances: {}'.format(tot_valid))
    print('Total invalid instances: {}'.format(tot_invalid))
    return ann_dict


# img = coco.loadImgs(ids=ids)[0]
# I = cv2.imread('/home/abhisek/Desktop/MSCOCO/val2017/' + img['file_name'])


def main():
    parser = argparse.ArgumentParser(description='Specify MSCOCO annotations and dataset paths')
    parser.add_argument('--train_anns', type=str,
            default='/home/abhisek/Desktop/MSCOCO/annotations_trainval2017/annotations/instances_train2017.json',
            help='COCO train annotations path')
    parser.add_argument('--val_anns', type=str,
            default='/home/abhisek/Desktop/MSCOCO/annotations_trainval2017/annotations/instances_val2017.json',
            help='COCO val annotations path')
    parser.add_argument('--train_imgids', type=str,
            default='/home/abhisek/Desktop/MSCOCO/train2017/',
            help='Training images directory path')
    parser.add_argument('--val_imgids', type=str,
            default='/home/abhisek/Desktop/MSCOCO/val2017/',
            help='Validation images directory path')
    args = parser.parse_args()

    anns_train = args.train_anns
    anns_val = args.val_anns
    train_ids = sorted(os.listdir(args.train_imgids))
    train_ids = [ids.lstrip('0') for ids in train_ids]
    train_ids = [int(ids.rstrip('.jpg')) for ids in train_ids]
    val_ids = sorted(os.listdir(args.val_imgids))
    val_ids = [ids.lstrip('0') for ids in val_ids]
    val_ids = [int(ids.rstrip('.jpg')) for ids in val_ids]

    coco_train = COCO(anns_train)
    train_dict = create_hulls(coco_train, train_ids)
    np.save('Train_hulls.npy', train_dict)
    coco_val = COCO(anns_val)
    val_dict = create_hulls(coco_val, val_ids)
    np.save('Val_hulls.npy', val_dict)

if __name__=='__main__':
    main()
