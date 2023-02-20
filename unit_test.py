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
import ipdb
from yolov6_obb.utils.envs import get_envs, select_device, set_random_seed
from yolov6_obb.data.datasets import *
from yolov6_obb.data.data_load import *
from yolov6_obb.utils.obb_utils import obb_vis
from tqdm import tqdm
import torch.distributed as dist
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--config', default='/home/crescent/YOLOv6-OBB/configs/repopt/yolov6s_opt.py',help='Model configuration')
    parser.add_argument('--img_size', default=640,help = 'Dataloader img size')
    parser.add_argument('--anno_file_name', default='/home/crescent/YOLOv6-OBB/train_DroneVehicle.txt')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    args = parser.parse_args()
    return args

def model_dataloader(args,visualize=True):
    args.rank, args.local_rank, args.world_size = get_envs()
    logger.info("The unit test dataset and dataloader")
    master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1
    device = select_device(args.device)
    if(visualize):
        args.batch_size = 1
    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))
    logger.info(device)
    args.rank, args.local_rank, args.world_size = get_envs()
    logger.info(f"training args are: {args}\n")
    if args.local_rank != -1: # if DDP mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        logger.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", \
                init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size)
    """
        def create_dataloader(
            anno_file_name,
            img_size,
            batch_size,
            rank = -1
            workers = 8,
            shuffle = False,
            data_dict = None,
):
    """
    train_loader = create_dataloader(args.anno_file_name, args.img_size[0], args.batch_size,
                                     args.local_rank,args.workers,shuffle=True)[0]
   
    if visualize:
        img, bboxes = iter(train_loader).__next__()
        img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img = img* 255.0
        img = img.astype(np.uint8)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
        print(bboxes)
        bboxes = bboxes[:,1:] #索引到当前的地址
        # [x,y,w,h,]
        obb_vis(img,bboxes)
    else:
        for imgs,bboxes in train_loader:
            logger.info(f"The img shape{imgs.shape}\n\
            The bboxes shape{bboxes.shape}")
        
        
    
 
def model_adaptation(args):
    logger.info("The unit test model adaption")
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
    model.train()
    x,feature_map = model(img)
    ipdb.set_trace()
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
        logger.info("The model has been successfully exported")
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, export_file)
    
if __name__ == "__main__":
    args = make_args()
    model_dataloader(args)
            