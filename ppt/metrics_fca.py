import torch
import numpy as np

class evaluation:
    def __init__(self):
        pass

    def IoU(self, boxes1, boxes2):
        """
        calculate IoU
        :param boxes1: (type:tensor) -> (4)
        :param boxes2: (type:tensor) -> (4)
        :return: IoU (type:double)
        """
        boxes1 = boxes1.reshape([1, -1])
        boxes2 = boxes2.reshape([1, -1])
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                  (boxes[:, 3] - boxes[:, 1]))
        # boxes1,boxes2,areas1,areas2的形状:
        # boxes1: (boxes1的数量,4),
        # boxes2: (boxes2的数量,4),
        # areas1: (boxes1的数量,),
        # areas2: (boxes2的数量,)
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        # inter_upperlefts, inter_lowerrights, inters的形状:
        # (boxes1的数量,boxes2的数量,2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        # inter_areas and union_areas的形状: (boxes1的数量,boxes2的数量)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas

    def symbol_recognition_recall(self, truth_boxes, predict_boxes, truth_labels, predict_labels,
                                  truth_keypoint_shapes, predict_keypoint_shapes, IoU_threshold, cls=None):
        """
        R = TP / (TP + FN)
        a score threshold of 0.7
        """
        shape_contrast = {}
        if cls is None:
            TP_and_FN = truth_boxes.shape[0]
        else:
            TP_and_FN = 0
            for i in truth_labels:
                if i == cls:
                    TP_and_FN += 1
        if TP_and_FN == 0:
            return -1
        TP = 0
        flag = np.zeros(predict_boxes.shape[0])
        # basic shape
        # print("truth_boxes:", truth_boxes)
        # print("predict_boxes:", predict_boxes)
        # print("truth_labels:", truth_labels)
        # print("predict_labels", predict_labels)
        # print("truth_keypoint_shapes", truth_keypoint_shapes)
        # print("predict_keypoint_shapes", predict_keypoint_shapes)
        for i, truth_box in enumerate(truth_boxes):
            if truth_labels[i] in [12, 13, 14]:
                continue
            for j, predict_box in enumerate(predict_boxes):
                IoU = self.IoU(predict_box, truth_box)
                if IoU > IoU_threshold and truth_labels[i] == predict_labels[j] and flag[j] == 0:
                    if cls is not None and truth_labels[i] == cls or cls is None:
                        TP += 1
                    flag[j] = 1
                    shape_contrast[i] = j
                    break
                shape_contrast[i] = -1
        # print("shape_contrast", shape_contrast)
        # arrow, line, double_arrow
        flag = np.zeros(predict_labels.shape[0])
        for i, truth_box in enumerate(truth_boxes):
            if truth_labels[i] not in [12, 13, 14]:
                continue
            for j, predict_box in enumerate(predict_boxes):
                IoU = 1  # self.IoU(predict_box, truth_box)
                if IoU > IoU_threshold and truth_labels[i] == predict_labels[j] and flag[j] == 0:
                    tidx = truth_keypoint_shapes[0]
                    for tidx in truth_keypoint_shapes:
                        if tidx[0] == i:
                            break
                    pidx = predict_keypoint_shapes[0]
                    for pidx in predict_keypoint_shapes:
                        if pidx[0] == j:
                            break
                    if tidx[1] == -1 or tidx[3] == -1:
                        continue
                    if predict_labels[j] == 3:
                        if shape_contrast[tidx[1]] == pidx[1] and \
                           shape_contrast[tidx[3]] == pidx[3]:
                            if cls is not None and truth_labels[i] == cls or cls is None:
                                TP += 1
                                flag[j] = 1
                            break
        return TP / TP_and_FN

    def symbol_recognition_precision(self, truth_boxes, predict_boxes, truth_labels, predict_labels,
                                     truth_keypoint_shapes, predict_keypoint_shapes, IoU_threshold, cls=None):
        return self.symbol_recognition_recall(predict_boxes, truth_boxes, predict_labels, truth_labels,
                                              predict_keypoint_shapes, truth_keypoint_shapes, IoU_threshold, cls)

    def symbol_recognition_F1(self, truth_boxes, predict_boxes, truth_labels, predict_labels,
                              truth_keypoint_shapes, predict_keypoint_shapes, cls=None):
        """
        F1 = 2 * P * R / (P + R)
        """
        recall = self.symbol_recognition_recall(truth_boxes, predict_boxes, truth_labels, predict_labels,
                                                truth_keypoint_shapes, predict_keypoint_shapes, 0.7, cls)
        precision = self.symbol_recognition_precision(truth_boxes, predict_boxes, truth_labels, predict_labels,
                                                      truth_keypoint_shapes, predict_keypoint_shapes, 0.7, cls)
        if precision < 0 or recall < 0:  # 其中一个是N/A
            return -1
        if precision + recall <= 1e-4:  # 二者都是0
            return 0
        return 2 * precision * recall / (precision + recall)

    def diagram_recognition_metrics(self, truth_boxes, predict_boxes, truth_labels, predict_labels,
                                    truth_keypoint_shapes, predict_keypoint_shapes, cls=None):
        """
        80% IoU
        """
        recall = self.symbol_recognition_recall(truth_boxes, predict_boxes, truth_labels, predict_labels,
                                                truth_keypoint_shapes, predict_keypoint_shapes, 0.8, cls)
        precision = self.symbol_recognition_precision(truth_boxes, predict_boxes, truth_labels, predict_labels,
                                                      truth_keypoint_shapes, predict_keypoint_shapes, 0.8, cls)
        if cls is None:
            tn = truth_boxes.shape[0]
            pn = predict_boxes.shape[0]
        else:
            tn = 0
            for i in truth_labels:
                if i == cls:
                    tn += 1
            pn = 0
            for i in predict_labels:
                if i == cls:
                    pn += 1
        if precision < 0 or recall < 0:  # 其中一个是N/A
            return -1
        if tn == pn and recall >= 0.99 and precision >= 0.99:
            return 1
        return 0

