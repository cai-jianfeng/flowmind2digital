# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: data_preprocess.py
@Date: 2022/12/29 12:52
@Author: caijianfeng
"""
import xml.etree.ElementTree as ET
import torch

class data_process:
    def __init__(self, category=None):
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

    def get_arrow_dicts(self, dataset_annotation_file):
        print('truth data annotation loading begin!')
        record = {}
        dataset_annotation_xml = ET.parse(dataset_annotation_file)  # acquire xml object
        root = dataset_annotation_xml.getroot()  # acquire root
        size = root.find('size')
        height, width = int(size.find('height').text), int(size.find('width').text)
        record['height'] = height
        record['width'] = width
        objects = root.findall('object')
        boxes = []
        arrow_keypoints = []
        classes = []
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
            boxes.append(bbox)
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
                keypoints = [[x_from, y_from, 2], [x_to, y_to, 2]]
            else:
                keypoints = [[0, 0, 0], [0, 0, 0]]
            arrow_keypoints.append(keypoints)
            cls = self.category[obj.find('name').text]
            classes.append(cls)
        print('truth data annotation load successfully!')
        return torch.tensor(boxes), torch.tensor(classes), torch.tensor(arrow_keypoints)

    def get_arrow_dicts_coco(self, dataset_annotation_file):
