# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: visio_connect_detectron2_keypoint.py
@Date: 2022/12/7 9:05
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

# In[1] predict bounding boxs and according labels & keypoints in test picture
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
arrow_metadata = MetadataCatalog.get('arrow_train')
print('register succeed!')

# eval model result
print('-----eval begin-----')
cfg = get_cfg()
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little for inference:
cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_keypoint_detection2_5000.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # [state, final state, text, arrow]
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.RETINANET.NUM_CLASSES = 7
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

predictor = DefaultPredictor(cfg)

demo_img_path = "./dataset_keypoint2/test/writer017_fc_001.png"

im = cv2.imread(demo_img_path)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=arrow_metadata,
               scale=0.8,
               instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
               )
out = v.draw_instance_predictions(outputs["instances"])  # draw result
cv2.imshow('eval', out.get_image()[:, :, ::-1])
cv2.waitKey()
cv2.destroyAllWindows()
outputs = outputs['instances']
bboxs = outputs.pred_boxes
print('bbox:', bboxs)
scores = outputs.scores
print('scores:', scores)
labels = outputs.pred_classes
print('labels:', labels)
keypoints = outputs.pred_keypoints
print('keypoints:', keypoints)
keypoints_heatmaps = outputs.pred_keypoint_heatmaps
# print('keypoint heatmaps:', keypoints_heatmaps)

# In[2] connect visio app to show predict result
from visio import visio_app

# category label <-> category name
category_opps = {0: 'Ellipse',
                 1: 'Hexagon',
                 # 2: 'Long ellipse',
                 2: 'Circle',
                 3: 'Rectangle',
                 4: 'Rhombus',
                 5: 'Trapezoid',
                 6: 'Triangle',
                 7: 'arrow'
                 }

# read data_shape_template_file_name
with open('./data_shape_name_contract.txt', 'r', encoding='utf-8') as f:
    data_shape_names = f.readlines()

# open visio app
app = visio_app()
appVisio = app.get_app(name='Visio.Application')
vdoc = app.add_doc(doc_name='Basic Diagram.vst',
                   app=appVisio)
page = app.add_page(doc=vdoc,
                    number=1)
for i, bbox in enumerate(bboxs):
    score = scores[i]
    label = category_opps[labels[i]]
    doc_name = 'BASIC_M.VSS'
    x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for data_shape_name in data_shape_names:
        data_shape_name = data_shape_name.strip()
        template_file_name, data_shape_name, data_shape_nameU = data_shape_name.split(';')
        if label == data_shape_nameU:
            doc_name = template_file_name
        break

    stn = app.get_template(template_name=doc_name,
                           app=appVisio)

    master = stn.Masters.ItemU(label)
    shp = page.Drop(master, x, y)
    shp.Text = 'text'
