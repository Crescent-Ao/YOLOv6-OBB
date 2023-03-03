import cv2
import numpy as np
import torch
import math
import time


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


def poly2rbox_le135_np(poly):
    """convert poly to rbox [-pi / 4, 3 * pi / 4]
    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        rbox: [cx, cy, w, h, angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt(
        (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])
    )
    edge2 = np.sqrt(
        (pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1])
    )

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    rbox_angle = 0
    if edge1 > edge2:
        rbox_angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        rbox_angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))

    rbox_angle = norm_angle(rbox_angle)

    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return [x_ctr, y_ctr, width, height, rbox_angle]


# 角度映射回0
def poly2rbox_oc_np(poly):
    """convert poly to rbox (0, pi / 2]
    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        rbox: [cx, cy, w, h, angle]
    """
    points = np.array(poly, dtype=np.float32).reshape((-1, 2))
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # using the new OpenCV Rotated BBox definition since 4.5.1
    # if angle < 0, opencv is older than 4.5.1, angle is in [-90, 0)
    if angle < 0:
        angle += 90
        w, h = h, w

    # convert angle to [0, 90)
    if angle == -0.0:
        angle = 0.0
    if angle == 90.0:
        angle = 0.0
        w, h = h, w

    angle = angle / 180 * np.pi
    return [cx, cy, w, h, angle]


def cal_line_length(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)
    )


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
        [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
        [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
    ]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = (
            cal_line_length(combinate[i][0], dst_coordinate[0])
            + cal_line_length(combinate[i][1], dst_coordinate[1])
            + cal_line_length(combinate[i][2], dst_coordinate[2])
            + cal_line_length(combinate[i][3], dst_coordinate[3])
        )
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combinate[force_flag]).reshape(8)


def rbox2poly_np(rboxes):
    """
    rboxes:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for i in range(len(rboxes)):
        x_ctr, y_ctr, width, height, angle = rboxes[i][:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        poly = get_best_begin_point_single(poly)
        polys.append(poly)
    polys = np.array(polys)
    return polys


def check_points_in_polys(points, polys):
    """Check whether point is in rotated boxes
    Args:
        points (tensor): (1, L, 2) anchor points
        polys (tensor): [B, N, 4, 2] gt_polys
        eps (float): default 1e-9
    Returns:
        is_in_polys (tensor): (B, N, L)
    """
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = polys.split(4, axis=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, 1]
    norm_ab = torch.sum(ab * ab, axis=-1)
    # [B, N, 1]
    norm_ad = torch.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_polys = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_polys


def check_points_in_rotated_boxes(points, boxes):
    """Check whether point is in rotated boxes
    Args:
        points (tensor): (1, L, 2) anchor points
        boxes (tensor): [B, N, 5] gt_bboxes
        eps (float): default 1e-9

    Returns:
        is_in_box (tensor): (B, N, L)
    """
    # [B, N, 5] -> [B, N, 4, 2]
    corners = box2corners(boxes)
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = corners.split([1,1,1,1], dim=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, L]
    norm_ab = torch.sum(ab * ab, axis=-1)
    # [B, N, L]
    norm_ad = torch.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_box = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_box


# rbox function implemented using paddle
def box2corners(box):
    """convert box coordinate to corners
    Args:
        box (Tensor): (B, N, 5) with (x, y, w, h, alpha) angle is in [0, 90)
    Returns:
        corners (Tensor): (B, N, 4, 2) with (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    B = box.shape[0]
    x, y, w, h, alpha = torch.split(box, [1, 1, 1, 1, 1], dim=-1)
    x4 = torch.Tensor([0.5, 0.5, -0.5, -0.5]).reshape(1, 1, 4).type_as(box)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.Tensor([-0.5, 0.5, 0.5, -0.5]).reshape((1, 1, 4)).type_as(box)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], axis=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.concat([cos, sin], axis=-1)
    row2 = torch.concat([-sin, cos], axis=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], axis=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.reshape([-1, 4, 2]), rot_T.reshape([-1, 2, 2]))
    rotated = rotated.reshape([B, -1, 4, 2])  # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def non_max_suppression_rotation(
    prediction,
    conf_thres=0.01,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
):
    """
    prediction: (tensor), with shape [N, 6 + num_classes], N is the number of bboxes.
       conf_thres: (float) confidence threshold.
       iou_thres: (float) iou threshold.
       classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
       agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
       multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
       max_det:(int), max number of output bboxes.
    """

    """
       predicition [bs,n_anchors,[x,y,w,h,angle,conf,nc_class]]
    """
    # 还是这么写[x,y,w,h,theta,conf,cls]
    nc = prediction.shape[2] - 6
    # nc = 1
    assert (
        0 <= conf_thres <= 1
    ), f"conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided."
    assert (
        0 <= iou_thres <= 1
    ), f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."
    pred_candidates = torch.logical_and(prediction[..., 5] > conf_thres, torch.max(prediction[..., 6:], axis=-1)[0] > conf_thres)  # candidates
    # xc = prediction[..., 5] > conf_thres
    xc = pred_candidates
    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    multi_label &= nc > 1
    t = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[
        0
    ]  # batch size
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue
        x[:, 6:] *= x[:, 5:6]
        box = x[:, :5]  # [x,y,w,h,theta]
        if multi_label:
            i, j = (x[:, 6:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 6, None], j[:, None].float()), 1)
            # box = box[i]
        else:
            conf, j = x[:, 6 : ].max(1, keepdim=True)
            inds = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float()), 1)[inds]
            # box 存在的问题
            # box = box[inds]
        if classes is not None:
            inds = (x[:, 6] == torch.tensor(classes, device=x.device)).any(1)
            # x = x[inds]
            # box = box[inds]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            keep_inds = x[:, 5].argsort(descending=True)[:max_nms]
            x = x[keep_inds]  # sort by confidence
            # box = box[keep_inds]
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 5]  # boxes (offset by class), scores
        boxes_xy = (box[:, :2].clone() + c).cpu().numpy()
        boxes_wh = box[:, 2:4].clone().cpu().numpy()
        boxes_angle = x[:, 4].clone().cpu().numpy()
        scores_for_cv2_nms = scores.cpu().numpy()
        boxes_for_cv2_nms = []
        for box_inds, _ in enumerate(boxes_xy):
            boxes_for_cv2_nms.append(
                (boxes_xy[box_inds], boxes_wh[box_inds], boxes_angle[box_inds])
            )
        i = cv2.dnn.NMSBoxesRotated(
            boxes_for_cv2_nms, scores_for_cv2_nms, conf_thres, iou_thres
        )
        i = torch.from_numpy(i).type(torch.LongTensor)
        i = i.squeeze(axis=-1)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded
    return output
