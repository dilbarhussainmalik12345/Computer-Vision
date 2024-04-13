import cv2
import os
import sys
import numpy as np
import json
import random
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer


def get_data_dicts(img_dir):
    classes = ["Blacky", "Niche"]

    json_file = os.path.join(img_dir, "cats-annotations.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        #print(v)

        if "regions" not in v: continue
        
        # Extract info from regions
        annos = v["regions"]
        objs = []

        for anno in annos:
            shape_attr = anno["shape_attributes"]
            px = shape_attr["all_points_x"]
            py = shape_attr["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            region_attr = anno["region_attributes"]
            current_class = region_attr["Class"]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(current_class),
                "iscrowd": 0
            }
            objs.append(obj)
        
        record["annotations"] = objs

        # Get info of the image
        filename = os.path.join(img_dir, current_class, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        dataset_dicts.append(record)
    return dataset_dicts


# # Verify data loading function
# def verify_dataloader():
#     from detectron2.data import DatasetCatalog, MetadataCatalog

#     for d in ["train", "val"]:
#         DatasetCatalog.register("cats_" + d, lambda d=d: get_data_dicts("cats/" + d))
#         MetadataCatalog.get("cats_" + d).set(thing_classes=["Blacky", "Niche"])

#     cats_metadata = MetadataCatalog.get("cats_train")

#     dataset_dicts = get_data_dicts("../data/train")

#     random.seed(1)
#     for d in random.sample(dataset_dicts, 3):
#         img = cv2.imread(d["file_name"])
#         visualizer = Visualizer(img[:, :, ::-1], metadata=cats_metadata, scale=0.5)
#         vis = visualizer.draw_dataset_dict(d)
#         #cv2.imshow(vis.get_image()[:, :, ::-1])
#         image_name = str(random.randint(0,100)) + ".jpg"
#         cv2.imwrite(image_name, vis.get_image()[:, :, ::-1])

# verify_dataloader()