""" Author and Maintainer : Abhisek Dey
    Master's Thesis Research
    This code tests MSCOCO convex hulls and diagonal intersection hulls
    NOT FOR COMMERCIAL USE"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

coco_train = COCO('/home/abhisek/Desktop/MSCOCO/annotations_trainval2017/annotations/instances_train2017.json')
new_dict = np.load('Train_hulls.npy', allow_pickle=True)
old_dict = np.load('Diagonal_hulls_train.npy', allow_pickle=True)
new_ids = list(new_dict[()].keys())
idx = np.random.randint(0, len(new_ids))
img = coco_train.loadImgs(ids=new_ids[idx])[0]
I = cv2.imread('/home/abhisek/Desktop/MSCOCO/train2017/' + img['file_name'])

new_anns = new_dict[()][new_ids[idx]][0]
old_anns = old_dict[()][new_ids[idx]][0]

for ann in new_anns:
    vertices = ann.reshape((-1,1,2)).astype('int32')
    I = cv2.polylines(I, [vertices], True, (0, 255, 255), thickness=2)

for ann in old_anns:
    vertices = ann.reshape((-1,1,2)).astype('int32')
    I = cv2.polylines(I, [vertices], True, (255, 255, 0), thickness=2)

plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.show()

# for idx in ann_dict[()]:
#     anns = ann_dict[()][idx][0]
#     diag_anns = diag_dict[()][idx][0]
#     for inst, ann in enumerate(anns):
#         act_ann = np.reshape(ann, (8,2))
#         diag_inst = diag_anns[inst].reshape((8,2))
#         plt.plot(act_ann[:,0], act_ann[:,1], 'k-')
#         plt.plot([act_ann[0,0],act_ann[-1,0]], [act_ann[0,1],act_ann[-1,1]], 'k-')
#         plt.plot(diag_inst[:, 0], diag_inst[:, 1], 'r-')
#         plt.plot([diag_inst[0, 0], diag_inst[-1, 0]], [diag_inst[0, 1], diag_inst[-1, 1]], 'r-')
#         plt.plot(diag_inst[:,0], diag_inst[:,1], 'b.')
#         plt.show()
#     exit(1)
