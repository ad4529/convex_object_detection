# FCOS for Convex Object Detection

Anchor-free based object detection methods have enabled looking towards a new paradigm of object detection where a bounding box should not be constrained to 4-corner points. In the box ground-truth instances, it is estimated that about 40% of the area inside a box is comprised of background information; with rotated objects and non-rectangular objects in the higer end of this statistic. In this Master's Thesis, I try to mitigate that problem by deriving the convex hull of all the objects from their corresponding segmentation masks and constraining them to 8 points. These 8 point polygons with their centroids serve as the ground-truth for training the modified FCOS. Instead of the original FCOS where each valid location regresses for the _left, right, top_ and _bottom_ offsets of the ground-truth, our network regresses for the _x_ and _y_ offsets of each of the 8 arbitrary points. Figure below shows a prediction example from the MSCOCO dataset.

| ![Convex FCOS Prediction](FCOS_detection1.png?raw=True "Prediction") |
|:--:|
| *Convex FCOS Prediction* |

## Creating Convex Annotations

All annotations were created from the MSCOCO_'17_ dataset. The folder `Custom_Annotations` contains the files to convert the COCO segmentation masks to 8-point Convex Hulls.

| ![Hull and Mask](Hull_vs_Centroid.png?raw=True "Hulls") |
|:--:|
| *The original convex hull (in Yellow) and the 8-point Hull* |

## Training

Currently, only ResNet-50 as a backbone has been tested. To train, make sure you have set the correct `.npy` file containing the convex hull annotations in `Convex_Object_detection/fcos_core/data/datasets/coco.py` and the correct path to the original COCO json annotations in `Convex\_Object\_detection/fcos_core/config/paths_catalog.py`. To train, use:
	python3 tools/train\_net.py --config-file configs/fcos/fcos\_R\_50\_FPN\_1x.yaml --skip-test DATALOADER.NUM_WORKERS 6 OUTPUT\_DIR _save path_

## Inference
	python3 tools/test\_net.py --config-file configs/fcos/fcos\_R\_50\_FPN\_1x.yaml MODEL.WEIGHT /MSCOCO/no\_convexity\_5C/_MODEL NAME.pth_ TEST.IMS\_PER\_BATCH _BATCH SIZE_




