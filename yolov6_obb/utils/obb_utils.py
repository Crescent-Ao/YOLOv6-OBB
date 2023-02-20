import cv2 
import torch
import numpy as np

def poly2obb_np(rbox:list):
    if rbox.shape[-1] == 9:
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


def obb2poly(obboxes,mode="xyxy"):
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
