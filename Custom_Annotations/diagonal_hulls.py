import numpy as np
from shapely.geometry import Polygon
import argparse
from multiprocessing import Pool
import psutil
import os
from tqdm import tqdm
from time import sleep


def get_diag_intersects(instance):
    angles = np.radians(np.arange(10,180,45))
    slope = np.tan(angles)
    poly = Polygon(instance)
    centroid = poly.centroid.coords
    centroid = centroid[0]
    consts = centroid[1] - (slope * centroid[0])
    ones = np.ones(len(consts))
    diag_eqns = np.vstack((-slope, ones, -consts)).T
    side_eqns = []
    for i in range(len(instance)):
        points = [instance[i], instance[(i+1)%8]]
        x_cord, y_cord = zip(*points)
        A = np.vstack([x_cord, np.ones(len(x_cord))]).T
        m,c = np.linalg.lstsq(A, y_cord, rcond=-1)[0]
        side_eqns.append([-m, 1, -c])
    side_eqns = np.asarray(side_eqns)
    side_eqns = np.reshape(side_eqns, (-1,3))
    intersections = np.zeros((8,2))

    for j in range(len(diag_eqns)):
        local_distances = []
        local_intersects = []
        for k in range(len(side_eqns)):
            x,y,z = np.cross(diag_eqns[j], side_eqns[k])
            if z != 0:
                local_intersects.append([x/z, y/z])
                local_distances.append(np.linalg.norm([x/z-centroid[0], y/z-centroid[1]]))
        sorted_intersects = [intersect for _,intersect in sorted(zip(local_distances, local_intersects))]
        act_intersects = sorted_intersects[:2]
        if act_intersects[0][0] >= centroid[0]:
            intersections[j] = act_intersects[0]
            intersections[j+4] = act_intersects[1]
        elif act_intersects[0][0] < centroid[0]:
            intersections[j+4] = act_intersects[0]
            intersections[j] = act_intersects[1]
        else:
            raise Exception("Something wrong with the intersections")

    assert np.any((intersections != 0)), 'Some diagonal intersects not found'
    intersections = np.reshape(intersections, 16)
    return intersections

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

    for idx, ann_dict in enumerate([val_anns]):
        new_dict = {}
        for ann in tqdm(ann_dict[()]):
            new_hulls = []
            instances = ann_dict[()][ann][0]
            invalids = ann_dict[()][ann][1]
            # pool = Pool(None, limit_cpu)
            # with pool as p:
            #     intersections = p.map(get_diag_intersects, instances)
            for instance in instances:
                intersections = get_diag_intersects(instance)
                new_hulls.append(intersections)
            new_dict[ann] = [new_hulls, invalids]
        if not idx:
            print('Saving diagonal training annotations')
            np.save(save_val, new_dict)
        else:
            print('Saving diagonal validation annotations')
            np.save(save_val)
    print('Done!')


if __name__=='__main__':
    main()
