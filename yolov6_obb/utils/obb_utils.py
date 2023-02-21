import cv2 
import torch
import numpy as np
import os
import mmcv.ops.box_iou_rotated as rotate_iou
def check_point_in_rotated_boxes(points, boxes):
      # [B, N, 5] -> [B, N, 4, 2]
    polys = obb2poly(boxes)
    points = points.unsqueeze(0)
    a, b, c, d = polys.split(4, axis=2)
    ab = b - a
    ad = d - a
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
    is_in_box = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (
        ap_dot_ad <= norm_ad)
    return is_in_box
def rotated_iou_similarity(box1,box2):
    """Calculate rotate iou of box1 and box2
    Args:
        box1 (Tensor): box with the shape [N,M1,5] 
        box2 (Tensor): box with the shape [N,M2,5]
    Return:
        iou (Tensor): box between box1 and box2 with shape [N,M1,M2]
    """
    rotated_ious = []
    for b1, b2 in zip(box1, box2):
        rotated_ious.append(rotate_iou(b1, b2))
    return torch.stack(rotate_iou,axis=0)
    
def box2corner(box):
      """convert box coordinate to corners
    Args:
        box (Tensor): (B, N, 5) with (x, y, w, h, alpha) angle is in [0, 90)
    Returns:
        corners (Tensor): (B, N, 4, 2) with (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    
def poly2obb_np(rbox:list):
    if rbox.shape[-1] == 9:
        print("")
        res = np.empty((*rbox.shape[:-1], 6))
        res[..., 5] = rbox[..., 8]
        rbox = rbox[..., :8].reshape(1, -1, 2).astype(np.float32)
    elif rbox.shape[-1] == 8:
        res = np.empty([*rbox.shape[:-1], 5])
        rbox = rbox.reshape(1, -1, 2).astype(np.float32) 
    else:
        raise NotImplementedError(" less than 8 which is not implemented")
    (x, y), (w, h), angle = cv2.minAreaRect(rbox)
    if w >= h:
        angle = -angle
    else:
        w, h = h, w
        angle = -90 - angle
    theta = angle / 180 * np.pi
    res[..., 0] = x 
    res[..., 1] = y
    res[..., 2] = w
    res[..., 3] = h
    res[..., 4] = theta
    
    return res[None,:]

def regular_theta(theta, mode='180', start=-np.pi/2):
    assert mode in ['360', '180']
    cycle = 2 * np.pi if mode == '360' else np.pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def obb2poly(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector1 = torch.cat(
        [w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat(
        [-h/2 * Sin, -h/2 * Cos], dim=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return torch.cat(
        [point1, point2, point3, point4], dim=-1)
def obb_vis(img:np.ndarray, rboxes:torch.Tensor):
    cv2.imwrite(os.path.join(os.getcwd(),'demo2.jpg'), img)
    obboxes = rboxes[:,1:]
    class_id = rboxes[:,0]
    poly_rotae = obb2poly(obboxes)
    for rbox,cls_id in zip(poly_rotae,class_id):
        rbox = np.array(rbox.reshape(1,-1,2)).astype(np.int32)
        cls_id = int(cls_id)
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        img = np.ascontiguousarray(img, dtype=np.int32)
        cv2.polylines(img, [rbox], True, color, 2)
    cv2.imwrite(os.path.join(os.getcwd(),'demo.jpg'), img)
    return img
        
        
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)