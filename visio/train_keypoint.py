# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: train_keypoint.py
@Date: 2022/12/5 16:50
@Author: caijianfeng
"""
import random

import cv2
# import some common libraries
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# In[1]: Some basic setup
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

category = {'circle': 0,
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
            }
keypoint_names = ['begin', 'end']
keypoint_flip_map = [('begin', 'end')]

# In[2] register dataset
def get_arrow_dicts(img_dir):
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
                'category_id': category[obj.find('name').text],
                'keypoints': keypoints
            }
            objs.append(obj_info)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == '__main__':
    for d in ['train', 'val']:
        DatasetCatalog.register('arrow_' + d, lambda d=d: get_arrow_dicts('dataset_arrow/' + d))
        MetadataCatalog.get('arrow_' + d).set(thing_classes=['{}'.format(categ) for categ in category.keys()])
    MetadataCatalog.get("arrow_train").keypoint_names = keypoint_names
    MetadataCatalog.get("arrow_train").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("arrow_train").evaluator_type = "coco"
    sketch_metadata = MetadataCatalog.get('arrow_train')
    print('register succeed!!')

    # In[] read img
    # # dataset_dicts = get_arrow_dicts('./dataset_arrow')
    # dataset_dicts = DatasetCatalog.get('arrow_train')
    # for d in tqdm(dataset_dicts):
    #     img = cv2.imread(d['file_name'])
    #     # print(d['file_name'])
    #     visualizer = Visualizer(img[:, :, ::-1],
    #                             metadata=sketch_metadata,
    #                             scale=1.0)
    #     out = visualizer.draw_dataset_dict(d)
    #     img_name = str(d['file_name'].split('\\')[-1])
    #     # cv2.imwrite('./dataset_arrow/flow_chart_ann/' + img_name, out.get_image()[:, :, ::-1])
    #     cv2.imshow('test', out.get_image()[:, :, ::-1])
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # In[3] train
    print('-----train begin-----')

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('arrow_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')  # # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000  # 1000 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # [state, final state, text, arrow]
    # cfg.MODEL.DEVICE = 'cpu'
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.RETINANET.NUM_CLASSES = 12
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    # Save the model
    # torch.save(trainer.model.state_dict(), './output/model_final_keypoint_detection.pth')
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save('model_final_keypoint_detection')
    print('-----train end------')

    # In[4]:eval model result
    print('-----eval begin-----')

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("arrow_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "arrow_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))  # use mAP
    # another equivalent way to evaluate the model is to use `trainer.test`
