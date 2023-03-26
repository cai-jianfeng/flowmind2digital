# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: truth_data.py
@Date: 2022/12/30 16:45
@Author: caijianfeng
"""
from predict import predict_mode
from data_preprocess import data_process
from visio import visio_app
import torch

# In[1] truth
predict_util = predict_mode()
data_util = data_process()
truth_boxes, truth_classes, truth_keypoints = data_util.get_arrow_dicts(
    dataset_annotation_file='./dataset_arrow/annotations/1119_0002.xml')

# In[2] y axis convert
max_y1 = torch.max(truth_boxes)
max_y2 = torch.max(truth_keypoints)
max_y = torch.max(max_y1, max_y2)
truth_boxes[:, 1] = max_y - truth_boxes[:, 1]
truth_boxes[:, 3] = max_y - truth_boxes[:, 3]
truth_keypoints[:, :, 1] = max_y - truth_keypoints[:, :, 1]
print('y axis convert successfully!')

# In[]
keypoints = predict_util.arrow_keypoint_convert(truth_boxes, truth_classes, truth_keypoints)
print('keypoints convert successfully!')

# In[3] visio
category = {0: 'Circle',
            1: 'Diamond',
            2: 'Rounded Rectangle',
            3: 'Hexagon',
            4: 'Parallelogram',
            5: 'Rectangle',
            6: 'Trapezoid',
            7: 'Triangle',
            8: 'Text',
            9: 'Arrow',
            10: 'Double_arrow',
            11: 'Line'
            }
app = visio_app()
visio = app.get_app(name='Visio.Application')
doc = app.add_doc(doc_name='Basic Diagram.vst')
page = app.add_page(doc=doc, number=1)
stn = app.get_template(template_name='BASIC_M.VSS')
# In[]
shps = []
for i, box in enumerate(truth_boxes):
    cls = int(truth_classes[i])
    if cls not in [8, 9, 10, 11]:
        master = app.choose_shape(shape_name=category[cls], template=stn)
        exec(
            f'shp{i} = app.add_shape(page=page, master=master, x=(box[0] + box[2]) / 50, y=(box[1] + box[3]) / 50)')

        exec(f'app.resize_shape(shp=shp{i}, w="{(box[2] - box[0]) * 3}pt", h="{(box[1] - box[3]) * 3}pt")')
        # shps.append(shp)
    # else:
    # shps.append('None')

# In[]
for i, dot_keypoints in enumerate(truth_keypoints):
    # print('keypoints:', keypoints.shape)
    cls = int(truth_classes[i])
    if cls in [8, 9, 10, 11]:
        for dot_keypoint in dot_keypoints:
            # print('keypoint:', keypoint.shape)
            master = app.choose_shape(shape_name='Circle', template=stn)
            shp = app.add_shape(page=page, master=master, x=f'{int(dot_keypoint[0]) // 25}', y=f'{int(dot_keypoint[1]) // 25}')
            app.resize_shape(shp=shp, w='50pt', h='50pt')

# In[4] connect
for keypoint in keypoints:
    exec(f'app.connect(page=page, shp1=shp{int(keypoint[1])}, shp2=shp{int(keypoint[3])}, shp1_point=int(keypoint[2]), shp2_point=int(keypoint[4]))')
