# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints
from .evaluation.coco.coco_visualizations import coco_visualization
from tqdm import tqdm
import numpy as np


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        #self.ids = sorted(self.ids)

        #print('Ids before: {}'.format(len(self.ids)))
        if remove_images_without_annotations: #Use training set annotations
            diag_dict = np.load('/MSCOCO/annotations_trainval2017/annotations/Train_hulls_3C.npy', allow_pickle=True)
        else: #Use validation set annotations
            diag_dict = np.load('/MSCOCO/annotations_trainval2017/annotations/Val_hulls_3C.npy', allow_pickle=True)    
        
        self.diag_dict = diag_dict
        new_ids = sorted(list(self.diag_dict[()].keys()))
        self.ids = new_ids
        # temp_ids = []

        # for idx in tqdm(new_ids):
        #     diags = self.diag_dict[()][idx]
        #     diags = [diag for diag in diags]
        #     diags = torch.as_tensor(diags, dtype=torch.float).reshape(-1, 10)
        #     if diags.size()[0] != 0:
        #         temp_ids.append(idx)
        # self.ids = temp_ids

        # filter images without detection annotations
        # if remove_images_without_annotations:
        #     ids = []
        #     for img_id in tqdm(self.ids):
        #         ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #         anno = self.coco.loadAnns(ann_ids)
        #         if has_valid_annotation(anno):
        #             ids.append(img_id)
        #     self.ids = ids

        print('Ids after: {}'.format(len(self.ids)))

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate([1,3,12])
        }
        # keys = list(self.json_category_id_to_contiguous_id.keys())
        # np.random.seed(28)
        # np.random.shuffle(keys)
        # self.json_category_id_to_contiguous_id = dict(zip(keys, self.json_category_id_to_contiguous_id.values()))

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        # self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)


        # filter crowd annotations
        # TODO might be better to add an extra field
        # anno = [obj for obj in anno if obj["iscrowd"] == 0]

        dict_key = self.ids[idx]
        diags = self.diag_dict[()][dict_key][0]
        invalids = self.diag_dict[()][dict_key][1]
        centers = self.diag_dict[()][dict_key][2]

        if not img.getbbox():
            img_data = img.convert("RGB")
            print(img_data)
            raise Exception('Image Error for key : {}'.format(dict_key))

        boxes = [obj["bbox"] for indx,obj in enumerate(anno) if indx not in invalids]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for indx,obj in enumerate(anno) if indx not in invalids]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        diags = torch.as_tensor(diags, dtype=torch.float).reshape(-1, 16)
        target.add_field("diagonals", diags)

        centers = torch.as_tensor(centers, dtype=torch.float).reshape(-1, 2)
        target.add_field("centers", centers)

        # img_id = self.id_to_img_map[idx]
        # img_data = self.coco.imgs[img_id]
        assert len(diags) == len(boxes), 'Mismatch in boxes and diags for id: {} with diag size {} and bbox size {} \n and boxes are: {}'.format(dict_key, diags.size(), boxes.size(), boxes)
        # coco_visualization(target, img_data, pre_train=True)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        #target = target.clip_to_image(remove_empty=True)

        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
