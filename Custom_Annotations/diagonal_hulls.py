import numpy as np
from shapely.geometry import Polygon
import argparse
from multiprocessing import Pool
import psutil
import os
from tqdm import tqdm
from time import sleep


def get_diag_intersects(instance):
    sleep(0.01)
    angles = np.radians(np.arange(10,180,45))
    slope = np.tan(angles)
    poly = Polygon(instance)
    centroid = poly.centroid.coords
    centroid = centroid[0]
    consts = centroid[1] - (slope * centroid[0])
    diag_eqns = np.vstack((slope, consts)).T

    return angles

def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(19)

def main():
    parser = argparse.ArgumentParser(description='Specify convex hull annotation paths')
    parser.add_argument('--train_anns', type=str, default='Train_hulls.npy')
    parser.add_argument('--val_anns', type=str, default='Val_hulls.npy')
    parser.add_argument('--save_path_train', type=str, default='Diagonal_hulls_train.npy')
    parser.add_argument('--save_path_val', type=str, default='Diagonal_hulls_val.npy')
    args = parser.parse_args()

    train_path = args.train_anns
    val_path = args.val_anns
    save_train = args.save_path_train
    save_val = args.save_path_val
    train_anns = np.load(train_path, allow_pickle=True)
    val_anns = np.load(val_path, allow_pickle=True)

    for idx, ann_dict in enumerate([train_anns, val_anns]):
        new_dict = {}
        for ann in ann_dict[()]:
            new_hulls = []
            instances = ann_dict[()][ann][0]
            invalids = ann_dict[()][ann][1]
            pool = Pool(None, limit_cpu)
            with pool as p:
                intersections = p.map(get_diag_intersects, instances)
            new_hulls.append(intersections)
            new_dict[ann] = [new_hulls, invalids]
        if not idx:
            print('Saving diagonal training annotations')
            np.save(save_train)
        else:
            print('Saving diagonal validation annotations')
            np.save(save_val)
    print('Done!')


if __name__=='__main__':
    main()
