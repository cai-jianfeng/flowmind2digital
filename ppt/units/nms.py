import torch
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

class Util:
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
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas

def work(predict_output):
    pred_boxes = predict_output['instances'].pred_boxes.tensor  # (num_box, 4)
    pred_classes = predict_output['instances'].pred_classes  # (num_box,)
    pred_keypoints = predict_output['instances'].pred_keypoints  # (num_box, 2, 3)
    pred_scores = predict_output['instances'].scores  # (num_box,)
    predict_output_instacne = predict_output['instances']
    # In[] NMS between classes
    util_tool = Util()
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

    # In[]
    # pred_boxes = torch.tensor(np.array(pred_boxes_NMS))
    # pred_classes = torch.tensor(np.array(pred_classes_NMS))
    # pred_keypoints = torch.tensor(np.array(pred_keypoints_NMS))
    pred_boxes = torch.tensor([item.detach().cpu().numpy() for item in pred_boxes_NMS])
    pred_classes = torch.tensor(pred_classes_NMS)
    pred_scores = torch.tensor(pred_scores_NMS)
    pred_keypoints = torch.tensor([item.detach().cpu().numpy() for item in pred_keypoints_NMS])
    pred_boxes = Boxes(pred_boxes)

    im_shape = predict_output['instances'].image_size
    ans = Instances(image_size=im_shape)
    ans.set('pred_boxes', pred_boxes)
    ans.set('pred_classes', pred_classes)
    ans.set('scores', pred_scores)
    ans.set('pred_keypoints', pred_keypoints)
    ans = {"instances": ans}
    return ans
