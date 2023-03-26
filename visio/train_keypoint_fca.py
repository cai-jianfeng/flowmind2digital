# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: train_keypoint_fca.py
@Date: 2022/12/6 19:46
@Author: caijianfeng
"""
import os

# import some common libraries
import numpy as np

# In[1]: Some basic setup
# import some common detectron2 utilities
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
    # In[4] train
    print('-----train begin-----')

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('arrow_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')  # # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 80000  # 80000 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # [state, final state, text, arrow]
    # cfg.MODEL.DEVICE = 'cpu'
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.RETINANET.NUM_CLASSES = 7
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    # Save the model
    # torch.save(trainer.model.state_dict(), './output/model_final_keypoint_detection.pth')
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save('model_final_fca_80k')
    print('-----train end------')
