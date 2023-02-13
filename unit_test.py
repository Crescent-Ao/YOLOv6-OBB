import argparse
import time
import sys
import os
import torch
import torch.nn as nn  
from loguru import logger
from yolov6_obb.models.yolo import *
from yolov6_obb.models.effidehead_obb import Detect
from yolov6_obb.layers.common import * 
from yolov6_obb.utils.events import LOGGER
from io import BytesIO
import onnx
import onnxsim
from yolov6_obb.utils.config import Config
import onnxruntime
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--config', default='/home/crescent/YOLOv6-OBB/configs/repopt/yolov6s_opt.py',help='Model configuration')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logger.info("model train begin")
    args = make_args()
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else "cpu")
    cfg = Config.fromfile(args.config)
    # 半精度仅仅针对GPU设备进行处理
    model = build_model(cfg, num_classes=15, device=device)
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
    img = torch.zeros(args.batch_size,3,*args.img_size).to(device)
    if args.half:
        img, model = img.half(),model.half()
    model.eval()
    output = model(img)
    logger.info("\n Starting to export onnx")
    export_file = '/home/crescent/YOLOv6-OBB/demo.onnx'
    with BytesIO() as f:
        torch.onnx.export(model, img, f, verbose=False, opset_version=13,
                          training = torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['feature_map']
                          )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model) 
        
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, export_file)
            