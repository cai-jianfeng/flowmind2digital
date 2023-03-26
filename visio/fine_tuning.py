# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: fine_tuning.py
@Date: 2023/2/15 16:56
@Author: caijianfeng
"""
import numpy as np
import os

import torch.cuda
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

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

# In[2] register dataset
register_coco_instances("arrow_train", {}, "./dataset_fine_tuning_fca/train.json", "./dataset_fine_tuning_fca/train")
register_coco_instances("arrow_test", {}, "./dataset_fine_tuning_fca/test.json", "./dataset_fine_tuning_fca/test")
MetadataCatalog.get('arrow_train').thing_classes = ['{}'.format(categ) for categ in category.keys()]
# MetadataCatalog.get("arrow_train").thing_dataset_id_to_contiguous_id = {1: 0, 2: 1, 3: 2, 4: 3}
MetadataCatalog.get("arrow_train").keypoint_names = keypoint_names
MetadataCatalog.get("arrow_train").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("arrow_train").evaluator_type = "coco"
arrow_metadata = MetadataCatalog.get('arrow_train')
print('register succeed!')

if __name__ == '__main__':
    # In[3] pre-train model cfg
    print('-----train begin-----')
    model_param = 'model_final_80k.pth'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('arrow_train',)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                     model_param)
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 10000 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'
    else:
        cfg.DATALOADER.NUM_WORKERS = 4
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.RETINANET.NUM_CLASSES = 7

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    # # In[]
    # trainer.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=8, bias=True).cuda() \
    #     if torch.cuda.is_available() else nn.Linear(in_features=1024, out_features=8, bias=True)  # num_classes + 1
    # trainer.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=28, bias=True).cuda() \
    #     if torch.cuda.is_available() else nn.Linear(in_features=1024, out_features=28, bias=True)  # num_classes * 4

    # In[]
    trainer.resume_or_load(resume=False)
    trainer.train()
    # Save the model
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save('model_final_fine_tuning_fca')
    print('-----train end------')

