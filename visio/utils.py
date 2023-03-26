# _*_ coding:utf-8 _*_
"""
@Software: detection2
@FileName: utils.py
@Date: 2023/1/28 14:21
@Author: caijianfeng
"""
import torch

class util:
    def __init__(self):
        pass

    def NMS(self, box1, box2, threshold):
        IoU_value = self.IoU(box1, box2)
        if IoU_value > threshold:
            return True
        else:
            return False

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