import sys
import pyds
import numpy as np
import torch
import torchvision

INPUT_H = 480  #defined in decode.h
INPUT_W = 640
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4



def xywh2xyxy(origin_h, origin_w, x,landmark):

        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h

        if r_h > r_w:
            y[:, 0] = x[:, 0] / r_w
            y[:, 2] = x[:, 2] / r_w
            y[:, 1] = (x[:, 1] - (INPUT_H - r_w * origin_h) / 2) / r_w
            y[:, 3] = (x[:, 3] - (INPUT_H - r_w * origin_h) / 2) / r_w
            
            landmark[:,0] = landmark[:,0]/r_w
            landmark[:,1] = (landmark[:,1] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,2] = landmark[:,2]/r_w
            landmark[:,3] = (landmark[:,3] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,4] = landmark[:,4]/r_w
            landmark[:,5] = (landmark[:,5] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,6] = landmark[:,6]/r_w
            landmark[:,7] = (landmark[:,7] - (INPUT_H - r_w * origin_h) / 2)/r_w
            landmark[:,8] = landmark[:,8]/r_w
            landmark[:,9] = (landmark[:,9] - (INPUT_H - r_w * origin_h) / 2)/r_w
        else:
            y[:, 0] = (x[:, 0] - (INPUT_W - r_h * origin_w) / 2) / r_h
            y[:, 2] = (x[:, 2] - (INPUT_W - r_h * origin_w) / 2) / r_h
            y[:, 1] = x[:, 1] /r_h
            y[:, 3] = x[:, 3] /r_h

            landmark[:,0] = (landmark[:,0] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,1] = landmark[:,1]/ r_h
            landmark[:,2] = (landmark[:,2] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,3] = landmark[:,3]/ r_h
            landmark[:,4] = (landmark[:,4] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,5] = landmark[:,5]/ r_h
            landmark[:,6] = (landmark[:,6] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,7] = landmark[:,7]/ r_h
            landmark[:,8] = (landmark[:,8] - (INPUT_W - r_h * origin_w) / 2)/r_h
            landmark[:,9] = landmark[:,9]/ r_h

        return y, landmark


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
    si = scores > CONF_THRESH
    boxes = boxes[si, :]
    scores = scores[si]

    landmark = landmark[si,:]

    origin_h = 720
    origin_w = 1280
    # Get boxes and landmark
    # boxes,landmark = xywh2xyxy(origin_h, origin_w, boxes,landmark)
    # Do nms
    indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
    result_boxes = boxes[indices, :].cpu()
    result_scores = scores[indices].cpu()
    result_landmark = landmark[indices].cpu()

    return result_landmark
    # return result_boxes, result_scores, result_landmark



