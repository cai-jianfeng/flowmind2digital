# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: predict_keypoint2.py
@Date: 2022/12/6 19:46
@Author: caijianfeng
"""
import cv2
import os
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# register dataset
category = {'connection': 0,
            'data': 1,
            'decision': 2,
            'process': 3,
            'terminator': 4,
            'text': 5,
            'arrow': 6
            }
keypoint_names = ['begin', 'end']
keypoint_flip_map = [('begin', 'end')]
register_coco_instances("arrow_train", {}, "./dataset_keypoint/train.json", "./dataset_keypoint/train")
MetadataCatalog.get('arrow_train').thing_classes = ['{}'.format(categ) for categ in category.keys()]
MetadataCatalog.get("arrow_train").keypoint_names = keypoint_names
MetadataCatalog.get("arrow_train").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("arrow_train").evaluator_type = "coco"
# arrow_metadata = MetadataCatalog.get('arrow_train')
print('register succeed!')

# eval model result
print('-----eval begin-----')
cfg = get_cfg()
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little for inference:
cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_keypoint_detectionpre.pth")  # path to the model we just trained
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # [state, final state, text, arrow]
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.RETINANET.NUM_CLASSES = 7
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

predictor = DefaultPredictor(cfg)
im = cv2.imread("./demo_keypoint3.png")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=arrow_metadata,
               scale=0.8,
               instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
               )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))  # draw result
cv2.imwrite('./output/eval_keypoint3.png', out.get_image()[:, :, ::-1])  # save result
print('-----eval end-----')
