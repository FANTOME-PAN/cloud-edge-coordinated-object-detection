import sys
import numpy as np
import torch
from data.helmet import HELMET_CLASSES
from layers.box_utils import jaccard
import pickle
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# detection size: 1 * num_clasess * top_k * 5(score, bbox)
def decode_raw_detection(detection, h, w):
    dets = [torch.tensor([]) for _ in range(detection.size(1) - 1)]
    for cls_idx in range(detection.size(1) - 1):
        cls_det = detection[0, cls_idx + 1]
        mask = (cls_det[:, 0] > 0.).unsqueeze(-1).expand_as(cls_det)
        cls_det = cls_det[mask].view(-1, 5)

        if cls_det.size(0) == 0:
            continue
        cls_det[:, 1] *= w
        cls_det[:, 3] *= w
        cls_det[:, 2] *= h
        cls_det[:, 4] *= h
        dets[cls_idx] = cls_det
    return dets


# step 1: find gt bbox with the same class and with max IOU for each detection bbox
# step 2: score = IOU * score
# step 3: ret = scores.mean()
def get_conf_gt(detection, h, w, annopath, classes=HELMET_CLASSES):
    num_classes = len(HELMET_CLASSES)
    dets = decode_raw_detection(detection, h, w)
    assert num_classes == len(dets)
    rec = parse_rec(annopath)
    bbgt = [torch.tensor([]) for _ in range(num_classes)]
    for cls_idx in range(num_classes):
        bbgt[cls_idx] = torch.tensor([x['bbox'] for x in rec if x['name'] == classes[cls_idx]], dtype=torch.float)
    bbdet = [dets[i][:, 1:].cpu() for i in range(len(dets))]

    cls_ious = [torch.tensor([]) for _ in range(num_classes)]
    for cls_idx in range(num_classes):
        # K * 4
        bb = bbdet[cls_idx]
        # N * 4
        gt = bbgt[cls_idx]
        iou = jaccard(gt, bb).t()
        cls_ious[cls_idx] = iou
    max_ious = [x.max(1)[0] for x in cls_ious]
    return cls_ious, max_ious



