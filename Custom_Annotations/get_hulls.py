""" Author and Maintainer : Abhisek Dey
    Master's Thesis Research
    This code converts MSCOCO segmentation annotations to it's corresponding convex hulls
    NOT FOR COMMERCIAL USE"""

import os
from pycocotools.coco import COCO
import numpy as np
import argparse
import cv2
from tqdm import tqdm
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


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
                    if len(contour) == 1:
                        contour = contour[0]
                        contour = cv2.convexHull(contour)
                        contour = np.squeeze(contour)
                        if len(contour) >= 8:
                            kmeans = KMeans(n_clusters=8, random_state=0).fit(contour)
                            centroids = kmeans.cluster_centers_
                            centroids = order_points(centroids)
                            area = Polygon(centroids).area
                            # Check if the convex hull is big enough
                            if area >= 5000:
                                if len(centroids) < 8:
                                    diff = 8 - len(centroids)
                                    centroids = np.vstack((centroids, np.tile(centroids[-1,:], (diff,1))))
                                new_hull = np.reshape(centroids, 16)
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
            else:
                invalids.append(pos)
                tot_invalid += 1

        if len(hulls):
            idx_ann = [hulls, invalids]
            ann_dict[idx] = idx_ann

    print('Total valid instances: {}'.format(tot_valid))
    print('Total invalid instances: {}'.format(tot_invalid))
    return ann_dict

# img = coco.loadImgs(ids=ids)[0]
# I = cv2.imread('/home/abhisek/Desktop/MSCOCO/val2017/' + img['file_name'])

# plt.plot(centroids[:, 0], centroids[:, 1], 'b')
# plt.plot([contour[0,0], contour[-1,0]], [contour[0,1], contour[-1,1]], 'b')
# img = coco.loadImgs(ids=ids)[0]
# I = cv2.imread('/home/abhisek/Desktop/MSCOCO/train2017/' + img['file_name'])
# centroids = centroids.astype('int32')
# centroids = centroids.reshape((-1,1,2))
# I = cv2.polylines(I, [centroids], True, (0,255,255), thickness=2)
# plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
# plt.show()

def order_points(centroids):
    hull = ConvexHull(centroids)
    vertices = hull.vertices
    centroids = centroids[vertices]
    start_index = np.argmax(centroids[:,0])
    centroids = np.concatenate((centroids[start_index:], centroids[0:start_index]), axis=0)
    return centroids

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
