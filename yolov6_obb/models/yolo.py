#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6_obb.layers.common import *
from yolov6_obb.utils.torch_utils import initialize_weights
from yolov6_obb.models.efficientrep import *
from yolov6_obb.models.reppan import *
from yolov6_obb.models.effidehead_obb_5 import Detect_OBB, build_effidehead_layer_OBB
# from yolov6_obb.models.effidehead_obb import Detect, build_effidehead_layer


class Model(nn.Module):
    '''yolov6_obb model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        #self.mode = config.training_mode
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, anchors, num_layers)

        # Init Detect head
        begin_indices = config.model.head.begin_indices
        out_indices_head = config.model.head.out_indices
        self.stride = self.detect.stride
        self.detect.i = begin_indices
        self.detect.f = out_indices_head
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        

        # x neck 之后的卷积
      #  fpn_feature, cls_score_list, reg_distri_list = x
       # output = torch.cat([cls_score_list, reg_distri_list], dim=-1)
        # 所以的这个preds 
        # print("output", output.shape)
        return x if export_mode is True else [x, featmaps]
     #   return output if export_mode is True else [output, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, anchors, num_layers):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_anchors = config.model.head.anchors
    use_reg_dfl = config.model.head.use_reg_dfl
    reg_max = config.model.head.reg_max
    use_angle_dfl = config.model.head.use_angle_dfl
    angle_max = config.model.head.angle_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    
    head_layers = build_effidehead_layer_OBB(channels_list,num_anchors,num_classes,reg_max=reg_max,angle_max=angle_max)

    head = Detect_OBB(num_classes, anchors, num_layers, head_layers=head_layers, use_reg_dfl=use_reg_dfl,use_angle_dfl=use_angle_dfl,reg_max=reg_max,angle_max=angle_max)

    return backbone, neck, head


def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    return model
