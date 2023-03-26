"""
created on:2022/10/28 13:11
@author:caijianfeng
"""
import distutils.core
import json
import os
import random
import sys

# import some common libraries
import cv2

dist = distutils.core.run_setup("./detectron2/setup.py")
sys.path.insert(0, os.path.abspath('./detectron2'))

import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# In[1]: 加载训练图片样例与预训练模型进行预测
# im = cv2.imread('./dataset/brace/1660923483556.png')
# cv2.imshow('test', im)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
#
# # look at the outputs.
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
#
# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow(out.get_image()[:, :, ::-1])
# cv2.waitKey()
# cv2.destroyAllWindows()

# In[2] 注册数据集
from detectron2.structures import BoxMode


def get_sketch_dicts(img_dir):
    json_file = os.path.join(img_dir, 'config.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # filename = os.path.join(img_dir, v['filename'])
        filename = v['img_path'][1:].replace('\\', '/')
        height, width = v['height'], v['width']
        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width
        bbox = [v['boy_y'][0], v['box_x'][0], v['boy_y'][1], v['box_x'][1]]
        category = v['category']
        obj = {
            'bbox': bbox,
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': category
        }
        objs = [obj]
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ['train', 'val']:
    DatasetCatalog.register('sketch_' + d, lambda d=d: get_sketch_dicts('./dataset'))
    MetadataCatalog.get('sketch_' + d).set(thing_classes=['{}'.format(i) for i in range(50)])
sketch_metadata = MetadataCatalog.get('sketch_train')

# dataset_dicts = get_sketch_dicts('./dataset')
# for d in random.sample(dataset_dicts, 3):
#     print(d['file_name'][1:])
#     img = cv2.imread(d['file_name'][1:])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=sketch_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('test', out.get_image()[:, :, ::-1])
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# In[3] 训练
print('-----train begin-----')
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ('sketch_train',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')  # # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 50  # has 50 class.
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print('-----train end------')

# In[4]:测试模型结果
print('-----eval begin-----')
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

i = 1
dataset_dicts = get_sketch_dicts("./dataset")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=sketch_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('eval', out.get_image()[:, :, ::-1])
    cv2.imwrite('eval_{}.png'.format(i), out.get_image()[:, :, ::-1])
    i += 1

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("sketch_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "sketch_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
