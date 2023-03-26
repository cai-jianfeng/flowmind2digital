# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: predict_keypoint3.py
@Date: 2022/11/28 9:38
@Author: caijianfeng
"""

# import some common libraries
import cv2
import os
import xml.etree.ElementTree as ET

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

category = {'circle/ellipse': 0,
            'hexagon': 1,
            'Long ellipse': 2,
            'rectangle': 3,
            'rhombus': 4,
            'trapezoid': 5,
            'triangle': 6
            }


# implement a function that returns the items in owning dataset
def get_sketch_dicts(img_dir):
    dataset_config = os.path.join(img_dir, 'config.txt')
    with open(dataset_config, 'r') as f:  # read all .xml file path
        dataset_annotation_files = f.readlines()

    dataset_dicts = []
    dataset_dir = os.path.join(img_dir, 'flow_chart')
    for idx, dataset_annotation_file in enumerate(dataset_annotation_files):
        dataset_annotation_file = dataset_annotation_file.strip()
        record = {}
        dataset_annotation_xml = ET.parse(dataset_annotation_file)  # acquire xml object
        root = dataset_annotation_xml.getroot()  # acquire root
        filename = os.path.join(dataset_dir, root.find('filename').text)  # read filename(from project root)
        size = root.find('size')
        height, width = int(size.find('height').text), int(size.find('width').text)
        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width
        objects = root.findall('object')
        objs = []
        for obj in objects:
            box = obj.find('bndbox')
            bbox = [int(box.find('xmin').text),
                    int(box.find('ymin').text),
                    int(box.find('xmax').text),
                    int(box.find('ymax').text)]
            obj_info = {
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': category[obj.find('name').text]
            }
            objs.append(obj_info)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# Register a Dataset
DatasetCatalog.register('sketch_train', lambda d='train': get_sketch_dicts('./dataset'))
# set category names(will show in picture)
MetadataCatalog.get('sketch_train').set(thing_classes=['{}'.format(categ) for categ in category.keys()])
# get dataset(train)
sketch_metadata = MetadataCatalog.get('sketch_train')

# set config info
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'))  # use COCO-Detection/faster_rcnn_R_50_C4_1x.yaml network
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_detection.pth")  # path to the model we trained
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # has 7 class.
cfg.MODEL.DEVICE = 'cpu'
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
# make output dir if it not exists
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# predict
print('-----predict begin-----')
# Inference should use the config with parameters that are used in training
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)  # generate a predictor via config info

im = cv2.imread('./demo_detection.jpg')  # read img we want to predict
outputs = predictor(im)  # predict img
print(outputs['instances'])
print(outputs['instances'].pred_boxes[0])
print(outputs['instances'].scores)
print(outputs['instances'].pred_classes)
v = Visualizer(im[:, :, ::-1],
               metadata=sketch_metadata,
               scale=0.5
               )  # init Visualizer to draw result
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))  # draw result
cv2.imwrite('./output/eval_detection.jpg', out.get_image()[:, :, ::-1])  # save result
