# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: test_keypoint_detection.py
@Date: 2022/12/5 16:25
@Author: caijianfeng
"""
# import some common libraries
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# In[1]
# read test img
im = cv2.imread('./test.png')
# Inference with a keypoint detection model
cfg = get_cfg()   # get a fresh new config
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow(out.get_image()[:, :, ::-1])
print(out)

# In[2]
import xml.etree.ElementTree as ET

f = ET.parse('./dataset_arrow/annotations/1119_0005.xml')
r = f.getroot()
objs = r.findall('object')
for obj in objs:
    p = obj.find('properties')
    if p:
        ks = p.findall('property')
        for k in ks:
            if 'y' in k.find('key').text:
                print(k.find('key').text)
