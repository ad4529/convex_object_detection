""" Author and Maintainer : Abhisek Dey
    Master's Thesis Research
    This code converts MSCOCO segmentation annotations to it's corresponding convex hulls
    NOT FOR COMMERCIAL USE"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

ann_dict = np.load('Val_hulls.npy', allow_pickle=True)
for idx in ann_dict[()]:
    anns = ann_dict[()][idx][0]
    for ann in anns:
        plt.plot(ann[:,0], ann[:,1], 'k-')
        plt.plot([ann[0,0],ann[-1,0]], [ann[0,1],ann[-1,1]], 'k-')
        plt.show()
    exit(1)
