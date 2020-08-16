from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .....structures.boxlist_ops import get_vertices_from_diags
from ... import coco

def coco_visualization(predictions, img_data=None, pre_train=False):

    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    # np.random.seed(28)
    # CATEGORIES = CATEGORIES[1:]
    # np.random.shuffle(CATEGORIES)
    # CATEGORIES.insert(0, "__background")
    if not pre_train:
        ann_file = "/MSCOCO/annotations_trainval2017/annotations/instances_val2017.json"
        root = "/MSCOCO/val2017/"

        for img_id,predict in predictions.items():
            img = coco.COCODataset(ann_file, root, False).get_img_info(img_id)
            I = cv2.imread(root + img['file_name'], 1)
            vertices = get_vertices_from_diags(predict)
            labels = predict.get_field("labels")
            # print(vertices)
            # print(labels)
            for indx,instance in enumerate(vertices):
                instance = instance.astype('int32')
                x_text = np.amin(instance[:,0], axis=0)
                y_text = np.amin(instance[:,1], axis=0) - 10
                instance = instance.reshape((-1,1,2))
                I = cv2.polylines(I, [instance], True, (0, 255, 255), thickness=2)
                cv2.putText(I, CATEGORIES[labels[indx]], (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

            plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
            plt.show()

    else:
        ann_file = "/MSCOCO/annotations_trainval2017/annotations/instances_train2017.json"
        root = "/MSCOCO/train2017/"
        boxes = predictions.bbox
        labels = predictions.get_field("labels")
        diagonals = predictions.get_field("diagonals")
        I = cv2.imread(root + img_data['file_name'], 1)
        vertices = get_vertices_from_diags(predictions)

        for indx,instance in enumerate(vertices):
            instance = instance.astype('int32')
            x_text = np.amin(instance[:,0], axis=0)
            y_text = np.amin(instance[:,1], axis=0) - 10
            instance = instance.reshape((-1,1,2))
            I = cv2.polylines(I, [instance], True, (0, 255, 255), thickness=2)
            cv2.putText(I, CATEGORIES[labels[indx]], (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

        plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
        plt.show()