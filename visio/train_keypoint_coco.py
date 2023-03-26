# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: train_keypoint_coco.py
@Date: 2022/12/6 15:24
@Author: caijianfeng
"""
# import some common libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET

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
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode

category = {'state': 0,
            'final state': 1,
            'text': 2,
            'arrow': 3
            }
keypoint_names = ['begin', 'end']
keypoint_flip_map = [('begin', 'end')]

# In[2] register dataset
register_coco_instances("arrow_train", {}, "./dataset_keypoint/train.json", "./dataset_keypoint/train")
register_coco_instances("arrow_test", {}, "./dataset_keypoint/test.json", "./dataset_keypoint/test")
MetadataCatalog.get('arrow_train').thing_classes = ['{}'.format(categ) for categ in category.keys()]
# MetadataCatalog.get("arrow_train").thing_dataset_id_to_contiguous_id = {1: 0, 2: 1, 3: 2, 4: 3}
MetadataCatalog.get("arrow_train").keypoint_names = keypoint_names
MetadataCatalog.get("arrow_train").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("arrow_train").evaluator_type = "coco"
arrow_metadata = MetadataCatalog.get('arrow_train')
print('register succeed!')

# In[3] plot picture
# dataset_dicts = DatasetCatalog.get("arrow_train")
#
# for d in random.sample(dataset_dicts, 5):
#     print(d["file_name"])
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=arrow_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('test', vis.get_image()[:, :, ::-1])
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# In[4] train
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
cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # [state, final state, text, arrow]
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.RETINANET.NUM_CLASSES = 4
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
# Save the model
# torch.save(trainer.model.state_dict(), './output/model_final_keypoint_detection.pth')
checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
checkpointer.save('model_final_keypoint_detection.pth')
print('-----train end------')
