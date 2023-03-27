# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: predict_flowchart.py
@Date: 2022/11/27 21:08
@Author: caijianfeng
"""
import xml.etree.ElementTree as ET
import json
import os
import cv2

categorys = {0: 'circle/ellipse',
            1: 'hexagon',
            2: 'Long ellipse',
            3: 'rectangle',
            4: 'rhombus/parallelogram',
            5: 'trapezoid',
            6: 'triangle'
            }

def get_predict_result(img_dir, predict_dir):
    json_file = os.path.join(img_dir, 'config.txt')
    with open(json_file, 'r') as f:
        imgs_anns_files = f.readlines()
    predict_file = os.path.join(predict_dir, 'coco_instances_results.json')
    with open(predict_file, 'r') as f:
        predict_anns = json.load(f)
    pre_img_id = 0
    pre = []
    for predict_ann in predict_anns:
        img_id = predict_ann['image_id']
        if img_id == pre_img_id:
            img_category = predict_ann['category_id']
            img_bbox = predict_ann['bbox']
            pre_info = {
                'bbox': img_bbox,
                'category': categorys[img_category]
            }
            pre.append(pre_info)
        else:
            img_ann_file = imgs_anns_files[pre_img_id].strip()
            img_ann = ET.parse(img_ann_file)
            root = img_ann.getroot()
            filename = root.find('filename').text
            filename = os.path.join(img_dir + '/flow_chart', filename)
            print('filename:', filename)
            objects = root.findall('object')
            objs = []
            for obj in objects:
                box = obj.find('bndbox')
                bbox = [int(box.find('xmin').text),
                        int(box.find('ymin').text),
                        int(box.find('xmax').text),
                        int(box.find('ymax').text)]
                category = obj.find('name').text
                obj_info = {
                    'bbox': bbox,
                    'category': category
                }
                objs.append(obj_info)
            # info = [objs, pre]
            info = [objs]
            save_filename = os.path.join(predict_dir + '/predict', str(pre_img_id) + '.jpg')
            print('save_filename:', save_filename)
            names = ['true bbox', 'predict bbox']
            draw_rectangle_by_point(img_file_path=filename,
                                    names=names,
                                    new_img_file_path=save_filename,
                                    bboxs=info)
            pre_img_id = img_id
            img_category = predict_ann['category_id']
            img_bbox = predict_ann['bbox']
            pre_info = {
                'bbox': img_bbox,
                'category': categorys[img_category]
            }
            pre = [pre_info]


# Plot the predicted box against the true box
def draw_rectangle_by_point(img_file_path, names, new_img_file_path, bboxs):
    image = cv2.imread(img_file_path)
    for i, info in enumerate(bboxs):
        for bbox_info in info:
            # print(bbox_info)
            bbox = bbox_info['bbox']
            category = bbox_info['category']
            first_point = (int(bbox[0]), int(bbox[1]))
            last_point = (int(bbox[2]), int(bbox[3]))

            print("top left corner:", first_point)
            print("lower right corner:", last_point)
            cv2.rectangle(image, first_point, last_point, (0, 255, 0), 1)  # Draw a box on the image
            cv2.putText(image, names[i] + '_' + category, first_point, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0),
                        thickness=1)  # Draw the name of the box above the rectangle
    # cv2.imshow('predict', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite(new_img_file_path, image)


if __name__ == '__main__':
    img_dir = './dataset'
    predict_dir = './output'
    get_predict_result(img_dir, predict_dir)
