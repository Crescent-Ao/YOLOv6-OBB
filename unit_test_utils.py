import torch.nn as nn
import torch
from loguru import logger
import numpy as np
import pandas as pd
import math
from yolov6_obb.utils.obb_utils import rotated_iou_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
def prepro_data(batch_data, device):
    images = batch_data[0].float().to(device, non_blocking=True)
    targets = batch_data[1].float().to(device)
    return images, targets
def get_optimizer(args, cfg, model):
    accumulate = max(1, round(64 / args.batch_size))
    cfg.solver.weight_decay *= args.batch_size * accumulate / 64
    optimizer = build_optimizer(cfg, model)
    return optimizer
def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer
def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf
def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        logger.error('unknown lr scheduler, use Cosine defaulted')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf
def update_optimizer(step,max_stepnum,epoch,optimizer,args,lf,cfg,scaler,warmup_stepnum,last_opt_step):
    curr_step = step + max_stepnum * epoch
    accumulate = max(1, round(64/args.batch_size))
    
    if curr_step <= warmup_stepnum:
            accumulate = max(1, np.interp(curr_step, [0, warmup_stepnum], [1, 64 / args.batch_size]).round())
            for k, param in enumerate(optimizer.param_groups):
                warmup_bias_lr = cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, warmup_stepnum], [warmup_bias_lr, param['initial_lr'] * lf(epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, warmup_stepnum], [cfg.solver.warmup_momentum, cfg.solver.momentum])
    if curr_step - last_opt_step >= accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            last_opt_step = curr_step
    return last_opt_step
        
        
class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.01, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 7]), x1, y1, x2, y2, conf, class, angle
            labels (Array[M, 6]), class, x, y, w, h, angle
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        # detections[:, :4] = xyxy2xywh(detections[:, :4])
        # iou = box_iou(labels[:, 1:5], detections[:, :4])
        boxes1 = torch.cat([detections[:, :4], detections[:, 6].view((-1, 1))], 1)
        boxes2 = torch.cat([labels[:, 1:5], labels[:, 5].view((-1, 1))], 1)
        # NOTE cv2 rotate angle
        iou = rotated_iou_similarity(boxes2, boxes1)
        # NOTE poly 8
        # iou = rotate_box_iou_poly(boxes2, boxes1)

        # iou = box_iou(labels[:, 1:5], detections[:, :4])
        iou = iou.type_as(detections)

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir="", names=()):
        try:
            import seaborn as sn
            import warnings
            from pathlib import Path
            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(
                    array,
                    annot=self.nc < 30,
                    annot_kws={"size": 8},
                    cmap="Blues",
                    fmt=".2f",
                    square=True,
                    xticklabels=names + ["background FP"] if labels else "auto",
                    yticklabels=names + ["background FN"] if labels else "auto",
                ).set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel("True")
            fig.axes[0].set_ylabel("Predicted")
            fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        except Exception as e:
            print(f"WARNING: ConfusionMatrix plot failure: {e}")

    def print(self):
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))
            
def rotate_box_iou(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 7]), x, y, w, h, angle, conf, class
        labels (Array[M, 6]), class, x, y, w, h, angle
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)

    # FIXME 不开转换即可，两次有损耗
    # detections[:, :4] = xyxy2xywh(detections[:, :4])
    # labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
    # boxes1 = torch.cat([detections[:, :4], detections[:, 6].view((-1, 1))], 1)

    boxes1 = detections[...,:5]
    boxes1_xywh = detections[..., :4]
    #boxes2 = torch.cat([labels[:, 1:5], labels[:, 5].view((-1, 1))], 1)
    boxes2_xywh = labels[...,1:5]
    boxes2_angle = labels[...,-1:]*180/np.pi
    boxes2 = torch.cat([boxes2_xywh,boxes2_angle],-1)
    
    # NOTE cv2 rotate angle
    iou = rotate_box_iou(boxes2.cpu().numpy(), boxes1.cpu().numpy())
    # NOTE poly 8
    # iou = rotate_box_iou_poly(boxes2, boxes1)
    import ipdb
    # iou = box_iou(boxes1_xywh, boxes2_xywh)
    iou = torch.from_numpy(iou).to(detections.device)
    # * iou >= iou_threshold and class_id 匹配
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, -1]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def ap_per_class(tp, conf, pred_cls, target_cls, AP_method="voc12", plot=True, save_dir=".", names=()):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], method=AP_method)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / "PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / "F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / "P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / "R_curve.png", names, ylabel="Recall")

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")

def compute_ap(recall, precision, method="voc12"):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    # mrec = np.concatenate(([0.0], recall, [1.]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    # NOTE: continous: VOC12， interp： COCO
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    elif method == "voc12":  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    else:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
    return ap, mpre, mrec

def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    
    
def plot_mc_curve(px, py, save_dir="mc_curve.png", names=(), xlabel="Confidence", ylabel="Metric"):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)