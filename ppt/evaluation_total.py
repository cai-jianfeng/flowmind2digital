# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: evaluation_total.py
@Date: 2023/1/12 13:27
@Author: liuhuanyu
"""
from predict import predict_mode
from metrics import evaluation
from data_preprocess import data_process
import os
import numpy as np

def calc4val(name):
    predict_output = predict_util.predict_flowchart(img_path='./hdflowmind/test/images/'+name+'.jpg')
    pred_boxes = predict_output['instances'].pred_boxes.tensor.cpu()  # (num_box, 4)
    pred_classes = predict_output['instances'].pred_classes.cpu()  # (num_box,)
    pred_keypoints = predict_output['instances'].pred_keypoints.cpu()  # (num_box, 2, 3)

    # In[3] arrow object
    convert_pred_keypoints = predict_util.arrow_keypoint_convert(pred_boxes, pred_classes, pred_keypoints)
    truth_boxes, truth_classes, truth_keypoints = data_util.get_arrow_dicts(
        dataset_annotation_file='./hdflowmind/test/annotations/'+name+'.xml')
    convert_truth_keypoints = predict_util.arrow_keypoint_convert(truth_boxes, truth_classes, truth_keypoints)
    # print("truth_boxes:", )
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
    ann = os.listdir('./hdflowmind/test/annotations')
    for i, name in enumerate(ann):
        ann[i] = name[:-4]

    predict_util = predict_mode()
    data_util = data_process()
    metrics_util = evaluation()
    n = np.zeros(4)
    ans = np.zeros(4)
    for name in ann:
        # print(name)
        val = calc4val(name)
        for i in range(4):
            if val[i] >= 0:
                n[i] += 1
                ans[i] += val[i]
        print(val, n)
    ans = ans / n
    print("total average recall in test set:", ans[0])
    print("total average precision in test set:", ans[1])
    print("total average F1 in test set:", ans[2])
    print("total average diagram_metrics in test set:", ans[3])
