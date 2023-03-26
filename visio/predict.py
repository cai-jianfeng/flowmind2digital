# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: predict.py
@Date: 2022/12/29 11:29
@Author: caijianfeng
"""
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F
import math
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes


# Some basic setup:

class predict_mode:
    def __init__(self, category=None, keypoint_names=None, keypoint_flip_map=None):
        self.category = {'circle': 0,
                         'diamonds': 1,
                         'long_oval': 2,
                         'hexagon': 3,
                         'parallelogram': 4,
                         'rectangle': 5,
                         'trapezoid': 6,
                         'triangle': 7,
                         'text': 8,
                         'arrow': 9,
                         'double_arrow': 10,
                         'line': 11
                         } if not category else category
        self.keypoint_names = ['begin', 'end'] if not keypoint_names else keypoint_names
        self.keypoint_flip_map = [('begin', 'end')] if not keypoint_flip_map else keypoint_flip_map

    def extra_setup(self):
        # Setup detectron2 logger
        setup_logger()

    def get_arrow_dicts(self, img_dir):
        dataset_config = os.path.join(img_dir, 'config.txt')
        with open(dataset_config, 'r') as f:  # read all .xml file path
            dataset_annotation_files = f.readlines()

        dataset_dicts = []
        dataset_dir = os.path.join(img_dir, 'flow_chart')
        for idx, dataset_annotation_file in enumerate(dataset_annotation_files):
            # print(dataset_annotation_file)
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
                # print(dataset_annotation_file)
                # print(obj.find('name'))
                if 'point' in obj.find('name').text:
                    continue
                box = obj.find('bndbox')
                bbox = [int(box.find('xmin').text),
                        int(box.find('ymin').text),
                        int(box.find('xmax').text),
                        int(box.find('ymax').text)]
                properties = obj.find('properties')
                if properties:
                    pro = properties.findall('property')
                    for keypoint in pro:
                        if keypoint.find('key').text == 'x_from':
                            x_from = int(keypoint.find('value').text)
                        elif keypoint.find('key').text == 'y_from':
                            y_from = height - int(keypoint.find('value').text)
                        elif keypoint.find('key').text == 'x_to':
                            x_to = int(keypoint.find('value').text)
                        elif keypoint.find('key').text == 'y_to':
                            y_to = height - int(keypoint.find('value').text)
                    keypoints = [x_from, y_from, 2, x_to, y_to, 2]
                else:
                    keypoints = [0, 0, 0, 0, 0, 0]
                obj_info = {
                    'iscrowd': 0,
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': self.category[obj.find('name').text],
                    'keypoints': keypoints
                }
                objs.append(obj_info)
            record['annotations'] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    def dataset_register(self, dataset_path):
        for i, d in enumerate(['train', 'eval']):
            DatasetCatalog.register('arrow_' + d, lambda d=d: self.get_arrow_dicts(dataset_path[i]))
            MetadataCatalog.get('arrow_' + d).set(thing_classes=['{}'.format(categ) for categ in self.category.keys()])
            # MetadataCatalog.get('arrow_' + d).set(thing_classes=['{}'.format(categ) for categ in range(12)])

        MetadataCatalog.get("arrow_train").keypoint_names = self.keypoint_names
        MetadataCatalog.get("arrow_train").keypoint_flip_map = self.keypoint_flip_map
        MetadataCatalog.get("arrow_train").evaluator_type = "coco"
        arrow_metadata = MetadataCatalog.get('arrow_train')
        print('register succeed!!')
        return arrow_metadata

    def predict_model(self, model_param, category):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                         model_param)  # path to the model we just trained
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category)
        cfg.MODEL.RETINANET.NUM_CLASSES = len(category)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(self.keypoint_names)
        cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((len(self.keypoint_names), 1), dtype=float).tolist()

        predictor = DefaultPredictor(cfg)
        return predictor

    def predict_flowchart(self, img_path, model_param):
        # eval model result
        print('-----eval begin-----')
        cfg = get_cfg()
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little for inference:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                         model_param)  # path to the model we just trained
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.category)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        cfg.MODEL.RETINANET.NUM_CLASSES = len(self.category)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(self.keypoint_names)
        cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((len(self.keypoint_names), 1), dtype=float).tolist()
        # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.9
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        predictor = DefaultPredictor(cfg)
        # print(predictor.model)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        print('-----eval end-----')
        return outputs

    def draw(self, img_path, output, save_path, arrow_metadata):
        if arrow_metadata:
            im = cv2.imread(img_path)
            v = Visualizer(im[:, :, ::-1],
                           metadata=arrow_metadata,
                           scale=1,
                           instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                           )
            im_shape = im.shape[:2]
            outputs = Instances(image_size=im_shape)
            outputs.set('pred_boxes', Boxes(output[0]))
            outputs.set('pred_classes', output[1])
            outputs.set('scores', output[2])
            outputs.set('pred_keypoints', output[3])

            out = v.draw_instance_predictions(outputs.to("cpu"))  # draw result
            # pred_boxes; pred_classes; pred_keypoints
            cv2.imwrite(save_path, out.get_image()[:, :, ::-1])  # save result
            print('draw succeed!')

    def arrow_keypoint_convert(self, pred_boxes, pred_classes, pred_keypoints):
        keypoints = []  # (num_arrow, 5)
        for i, pred_box in enumerate(pred_boxes):
            if pred_classes[i] in [9, 10, 11]:
                points = [i]  # (arrow_id, point_from_shape_id, point_from_shape_point, point_to_shape_id, point_to_shape_point)
                for pred_keypoint in pred_keypoints[i]:
                    min_distance = math.inf
                    min_point = [-1, -1]
                    for j, box in enumerate(pred_boxes):
                        if pred_classes[j] not in [8, 9, 10, 11]:
                            point, distance = self.shapes_connect(keypoint=pred_keypoint[:2], cls=pred_classes[j], box=box)
                            if distance < min_distance:
                                min_point = [j, point]
                                min_distance = distance
                    points.append(min_point[0])
                    points.append(min_point[1])
                keypoints.append(points)
        return np.array(keypoints)

    def shapes_connect(self, keypoint, cls, box):
        if cls == -1:  # oval
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0], (box[1] + box[3]) / 2]
            right = [box[2], (box[1] + box[3]) / 2]
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            points = torch.tensor([up, center, left, down, right])
        if cls == 0:  # circle
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0], (box[1] + box[3]) / 2]
            right = [box[2], (box[1] + box[3]) / 2]
            ratio = (2**0.5 + 1) / (2 * 2**0.5)
            diameter = (box[1] + box[3] - box[0] - box[2]) / 2
            ratio *= diameter
            top_left = [box[0] + ratio, box[1] + ratio]
            lower_left = [box[0] + ratio, box[3] - ratio]
            top_right = [box[2] - ratio, box[1] + ratio]
            lower_right = [box[2] - ratio, box[3] - ratio]
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            points = torch.tensor([up, center, left, down, right, lower_left, lower_right, top_right, top_left])
        if cls == 1:  # diamonds
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0], (box[1] + box[3]) / 2]
            right = [box[2], (box[1] + box[3]) / 2]
            points = torch.tensor([down, right, up, left])
        if cls == 2:  # long oval
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0], (box[1] + box[3]) / 2]
            right = [box[2], (box[1] + box[3]) / 2]
            top_left = [box[0], box[1]]
            lower_left = [box[0], box[3]]
            top_right = [box[2], box[1]]
            lower_right = [box[2], box[3]]
            points = torch.tensor([left, right, down, up, top_left,
                                  top_right, lower_left, lower_right])
        if cls == 3:  # hexagon
            up1 = [box[0] + (box[2] - box[0]) / 3, box[1]]
            up2 = [box[2] - (box[2] - box[0]) / 3, box[1]]
            down1 = [box[0] + (box[2] - box[0]) / 3, box[3]]
            down2 = [box[2] - (box[2] - box[0]) / 3, box[3]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0], (box[1] + box[3]) / 2]
            right = [box[2], (box[1] + box[3]) / 2]
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            points = torch.tensor([center, right, down1, down2, left, up1, up2])
        if cls == 4:  # parallelogram
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0] + (box[2] - box[0]) / 10, (box[1] + box[3]) / 2]
            right = [box[2] - (box[2] - box[0]) / 10, (box[1] + box[3]) / 2]
            points = torch.tensor([left, right, down, up])
        if cls == 5:  # rectangle
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0], (box[1] + box[3]) / 2]
            right = [box[2], (box[1] + box[3]) / 2]
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            points = torch.tensor([down, right, up, left, center])
        if cls == 6:  # trapezoid
            up = [(box[0] + box[2]) / 2, box[1]]
            down = [(box[0] + box[2]) / 2, box[3]]
            left = [box[0] + (box[2] - box[0]) / 10, (box[1] + box[3]) / 2]
            right = [box[2] - (box[2] - box[0]) / 10, (box[1] + box[3]) / 2]
            points = torch.tensor([left, right, down, up])
        if cls == 7:  # triangle
            up = [(box[0] + box[2]) / 2, box[1]]
            lower_left = [box[0], box[3]]
            lower_right = [box[2], box[3]]
            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            points = torch.tensor([center, lower_left, lower_right, up])
        # print(keypoint.size())
        # print(points.size())
        distances = F.pairwise_distance(keypoint, points, p=2)
        point = torch.argmin(distances)
        distance = torch.min(distances)
        return point, distance
