# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: predict_keypoint.py
@Date: 2022/12/6 19:27
@Author: caijianfeng
"""
from predict import predict_mode
from visio import visio_app
import torch
import shutil
import os
from utils import util

# In[1] predict util
predict_util = predict_mode()
predict_util.extra_setup()

# In[2] predict
try:
    arrow_metadata = predict_util.dataset_register(dataset_path=['./dataset_arrow', './dataset_arrow'])
except:
    print('Dataset is already registered!')
model_param = 'model_final_80k.pth'
predict_output = predict_util.predict_flowchart(img_path='./demo.jpg', model_param=model_param)
# pred_boxes; pred_classes; pred_keypoints
pred_boxes = predict_output['instances'].pred_boxes.tensor  # (num_box, 4)
pred_classes = predict_output['instances'].pred_classes  # (num_box,)
pred_keypoints = predict_output['instances'].pred_keypoints  # (num_box, 2, 3)
pred_scores = predict_output['instances'].scores  # (num_box,)
predict_output_instacne = predict_output['instances']

# In[3] NMS between classes
util_tool = util()
pred_boxes_NMS = []
pred_classes_NMS = []
pred_keypoints_NMS = []
pred_scores_NMS = []
for i, pred_box in enumerate(pred_boxes):
    flag = 0
    for j, pred_box_NMS in enumerate(pred_boxes_NMS):
        if util_tool.NMS(pred_box, pred_box_NMS, threshold=0.8):
            flag = 1
            if pred_scores[i] > pred_scores_NMS[j]:
                pred_boxes_NMS.pop(j)
                pred_classes_NMS.pop(j)
                pred_keypoints_NMS.pop(j)
                pred_scores_NMS.pop(j)
                pred_boxes_NMS.append(pred_box)
                pred_classes_NMS.append(pred_classes[i])
                pred_keypoints_NMS.append(pred_keypoints[i])
                pred_scores_NMS.append(pred_scores[i])
                break
    if flag == 0:
        pred_boxes_NMS.append(pred_box)
        pred_classes_NMS.append(pred_classes[i])
        pred_keypoints_NMS.append(pred_keypoints[i])
        pred_scores_NMS.append(pred_scores[i])

# In[4] integrate into Instance sub-object
# pred_boxes = torch.tensor(np.array(pred_boxes_NMS))
# pred_classes = torch.tensor(np.array(pred_classes_NMS))
# pred_keypoints = torch.tensor(np.array(pred_keypoints_NMS))
pred_boxes = torch.tensor([item.detach().numpy() for item in pred_boxes_NMS])
# pred_classes = torch.tensor([item.detach().numpy() for item in pred_classes_NMS])
pred_classes = torch.tensor(pred_classes_NMS)
pred_scores = torch.tensor(pred_scores_NMS)
pred_keypoints = torch.tensor([item.detach().numpy() for item in pred_keypoints_NMS])
# pred_boxes = torch.tensor(pred_boxes_NMS)
# pred_classes = torch.tensor(pred_classes_NMS)
# pred_keypoints = torch.tensor(pred_keypoints_NMS)

# In[5] draw predict result
# pred_boxes = Boxes(pred_boxes)
predict_util.draw(img_path='./demo_complex.jpg',
                  output=[pred_boxes, pred_classes, pred_scores, pred_keypoints],
                  save_path='./output/eval_complex.jpg',
                  arrow_metadata=arrow_metadata)

# In[6] y axis convert
max_y1 = torch.max(pred_boxes)
max_y2 = torch.max(pred_keypoints)
max_y = torch.max(max_y1, max_y2)
pred_boxes[:, 1] = max_y - pred_boxes[:, 1]
pred_boxes[:, 3] = max_y - pred_boxes[:, 3]
pred_keypoints[:, :, 1] = max_y - pred_keypoints[:, :, 1]
print('y axis convert successfully!')

# In[7] arrow object
keypoints = predict_util.arrow_keypoint_convert(pred_boxes, pred_classes, pred_keypoints)
print('keypoints convert successfully!')

# In[8] visio
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
for i, box in enumerate(pred_boxes):
    cls = int(pred_classes[i])
    if cls not in [8, 9, 10, 11]:
        master = app.choose_shape(shape_name=category[cls], template=stn)
        exec(
            f'shp{i} = app.add_shape(page=page, master=master, x=(box[0] + box[2]) / 50, y=(box[1] + box[3]) / 50)')

        exec(f'app.resize_shape(shp=shp{i}, w="{(box[2] - box[0]) * 3}pt", h="{(box[1] - box[3]) * 3}pt")')

for i, dot_keypoints in enumerate(pred_keypoints):
    # print('keypoints:', keypoints.shape)
    cls = int(pred_classes[i])
    if cls in [9, 10, 11]:
        for dot_keypoint in dot_keypoints:
            # print('keypoint:', keypoint.shape)
            master = app.choose_shape(shape_name='Circle', template=stn)
            shp = app.add_shape(page=page, master=master, x=f'{int(dot_keypoint[0]) // 25}', y=f'{int(dot_keypoint[1]) // 25}')
            app.resize_shape(shp=shp, w='50pt', h='50pt')
# In[9] connect
for keypoint in keypoints:
    exec(f'app.connect(page=page, shp1=shp{int(keypoint[1])}, shp2=shp{int(keypoint[3])}, shp1_point=int(keypoint[2]), shp2_point=int(keypoint[4]))')

# In[10] save
app.save(doc, r'D:\softward\detection2\demo.vsdx')
app.close(doc, visio)

shutil.rmtree(r'C:\Users\86199\AppData\Local\Temp\gen_py\3.8')
os.mkdir(r'C:\Users\86199\AppData\Local\Temp\gen_py\3.8')
