# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: visio_connect_detectron2.py
@Date: 2022/11/29 23:27
@Author: caijianfeng
"""
# In[0] choose test picture
test_pic = './demo.jpg'

# In[1] predict bounding boxs and according labels in test picture
# import some common libraries
import cv2
import os
import xml.etree.ElementTree as ET

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode

category = {'circle/ellipse': 0,
            'hexagon': 1,
            'Long ellipse': 2,
            'rectangle': 3,
            'rhombus/parallelogram': 4,
            'trapezoid': 5,
            'triangle': 6
            }


# implement a function that returns the items in owning dataset
def get_sketch_dicts(img_dir):
    dataset_config = os.path.join(img_dir, 'config.txt')
    with open(dataset_config, 'r') as f:  # read all .xml file path
        dataset_annotation_files = f.readlines()

    dataset_dicts = []
    dataset_dir = os.path.join(img_dir, 'flow_chart')
    for idx, dataset_annotation_file in enumerate(dataset_annotation_files):
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
            box = obj.find('bndbox')
            bbox = [int(box.find('xmin').text),
                    int(box.find('ymin').text),
                    int(box.find('xmax').text),
                    int(box.find('ymax').text)]
            obj_info = {
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': category[obj.find('name').text]
            }
            objs.append(obj_info)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# Register a Dataset
DatasetCatalog.register('sketch_train', lambda d='train': get_sketch_dicts('./dataset'))
# set category names(will show in picture)
MetadataCatalog.get('sketch_train').set(thing_classes=['{}'.format(categ) for categ in category.keys()])
# get dataset(train)
sketch_metadata = MetadataCatalog.get('sketch_train')

# set config info
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'))  # use COCO-Detection/faster_rcnn_R_50_C4_1x.yaml network
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we trained
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # has 7 class.
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
# make output dir if it not exists
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# predict
print('-----predict begin-----')
# Inference should use the config with parameters that are used in training
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)  # generate a predictor via config info

im = cv2.imread(test_pic)  # read img we want to predict
outputs = predictor(im)  # predict img
outputs = outputs['instances']
bboxs = outputs.pred_boxes
scores = outputs.scores
labels = outputs.pred_classes

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
                 6: 'Triangle'
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
