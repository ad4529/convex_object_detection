""" Author and Maintainer : Abhisek Dey
    Master's Thesis Research
    This code tests MSCOCO convex hulls and diagonal intersection hulls
    NOT FOR COMMERCIAL USE"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

ann_dict = np.load('Val_hulls.npy', allow_pickle=True)
diag_dict = np.load('Diagonal_hulls_val.npy', allow_pickle=True)
for idx in ann_dict[()]:
    anns = ann_dict[()][idx][0]
    diag_anns = diag_dict[()][idx][0]
    for inst, ann in enumerate(anns):
        act_ann = np.reshape(ann, (8,2))
        diag_inst = diag_anns[inst].reshape((8,2))
        plt.plot(act_ann[:,0], act_ann[:,1], 'k-')
        plt.plot([act_ann[0,0],act_ann[-1,0]], [act_ann[0,1],act_ann[-1,1]], 'k-')
        plt.plot(diag_inst[:, 0], diag_inst[:, 1], 'r-')
        plt.plot([diag_inst[0, 0], diag_inst[-1, 0]], [diag_inst[0, 1], diag_inst[-1, 1]], 'r-')
        plt.plot(diag_inst[:,0], diag_inst[:,1], 'b.')
        plt.show()
    exit(1)
