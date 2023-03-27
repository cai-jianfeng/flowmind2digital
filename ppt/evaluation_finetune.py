# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: evaluation_total.py
@Date: 2023/1/12 13:27
@Author: liuhuanyu
"""
from predict_finetune import predict_mode
from metrics_fca import evaluation
import numpy as np
import json
import torch

def calc4val(name, truth_boxes, truth_classes, truth_keypoints):
    predict_output = predict_util.predict_flowchart(img_path='./hdBPMN2021/test/' + name)
    pred_boxes = predict_output['instances'].pred_boxes.tensor.cpu()  # (num_box, 4)
    pred_classes = predict_output['instances'].pred_classes.cpu()  # (num_box,)
    pred_keypoints = predict_output['instances'].pred_keypoints.cpu()  # (num_box, 2, 3)
    convert_pred_keypoints = predict_util.arrow_keypoint_convert(pred_boxes, pred_classes, pred_keypoints)
    convert_truth_keypoints = predict_util.arrow_keypoint_convert(truth_boxes, truth_classes, truth_keypoints)
    # print(pred_boxes)
    # print(pred_classes)
    # print(convert_pred_keypoints)
    # print(truth_boxes)
    # print(truth_classes)
    # print(convert_truth_keypoints)
    recall = metrics_util.symbol_recognition_recall(truth_boxes=truth_boxes,
                                                    predict_boxes=pred_boxes,
                                                    truth_labels=truth_classes,
                                                    predict_labels=pred_classes,
                                                    truth_keypoint_shapes=convert_truth_keypoints,
                                                    predict_keypoint_shapes=convert_pred_keypoints,
                                                    IoU_threshold=0.7)
    precision = metrics_util.symbol_recognition_precision(truth_boxes=truth_boxes,
                                                          predict_boxes=pred_boxes,
                                                          truth_labels=truth_classes,
                                                          predict_labels=pred_classes,
                                                          truth_keypoint_shapes=convert_truth_keypoints,
                                                          predict_keypoint_shapes=convert_pred_keypoints,
                                                          IoU_threshold=0.7)
    F1 = metrics_util.symbol_recognition_F1(truth_boxes=truth_boxes,
                                            predict_boxes=pred_boxes,
                                            truth_labels=truth_classes,
                                            predict_labels=pred_classes,
                                            truth_keypoint_shapes=convert_truth_keypoints,
                                            predict_keypoint_shapes=convert_pred_keypoints,)
    diagram_metrics = metrics_util.diagram_recognition_metrics(truth_boxes=truth_boxes,
                                                               predict_boxes=pred_boxes,
                                                               truth_labels=truth_classes,
                                                               predict_labels=pred_classes,
                                                               truth_keypoint_shapes=convert_truth_keypoints,
                                                               predict_keypoint_shapes=convert_pred_keypoints,)
    return np.stack([recall, precision, F1, diagram_metrics], axis=0)


if __name__ == '__main__':
    with open('./hdBPMN2021/test.json', 'r', encoding='utf8')as fp:
        ftot = json.load(fp)
        predict_util = predict_mode()
        metrics_util = evaluation()
        n = np.zeros(4)
        ans = np.zeros(4)
        ann = ftot["annotations"]
        m = len(ann)
        j = 0
        for i, aid in enumerate(ftot["images"]):
            name = aid["file_name"]
            bboxes = []
            clses = []
            kpts = []
            while j < m and ann[j]["image_id"] == i:
                bbox = ann[j]["bbox"]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bboxes.append(bbox)
                clses.append(ann[j]["category_id"])
                kpts.append(ann[j]["keypoints"])
                j += 1
            bboxes = torch.tensor(bboxes)
            clses = torch.tensor(clses)
            kpts = torch.tensor(kpts)
            kpts = kpts.view([-1, 2, 3])
            val = calc4val(name, bboxes, clses, kpts)
            for ii in range(4):
                if val[ii] >= 0:
                    n[ii] += 1
                    ans[ii] += val[ii]
            print(val, n)
        ans = ans / n
        print("total average recall in test set:", ans[0])
        print("total average precision in test set:", ans[1])
        print("total average F1 in test set:", ans[2])
        print("total average diagram_metrics in test set:", ans[3])
