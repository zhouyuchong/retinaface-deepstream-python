'''
Author: zhouyuchong
Date: 2023-07-12 09:28:07
Description: 
LastEditors: zhouyuchong
LastEditTime: 2023-07-12 10:35:35
'''
import sys
import pyds
import numpy as np
import torch
import torchvision

CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4

def parse_objects_from_tensor_meta(layer):
    num_detection = 0
    if layer.buffer:
        num_detection = int(pyds.get_detections(layer.buffer, 0))
        # print(num_detection)
    
    num = num_detection * 15 + 1

    output = []
    for i in range(num):
        output.append(pyds.get_detections(layer.buffer, i))
    
    pred = np.reshape(output[1:], (-1, 15))[:num, :]

    pred = torch.Tensor(pred).cuda()
    # Get the boxes
    boxes = pred[:, :4]
    # Get the scores
    scores = pred[:, 4]
    # Get the landmark
    landmark = pred[:,5:15]
    # Choose those boxes that score > CONF_THRESH
    si = scores >= CONF_THRESH
    boxes = boxes[si, :]
    scores = scores[si]

    landmark = landmark[si,:]

    # Do nms
    indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
    # result_boxes = boxes[indices, :].cpu()
    # result_scores = scores[indices].cpu()
    result_landmark = landmark[indices].cpu()

    return result_landmark
    # return result_boxes, result_scores, result_landmark



