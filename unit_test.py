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
from yolov6_obb.models.loss_obb import ComputeLoss
from tqdm import tqdm
import torch.distributed as dist
from torch.cuda import amp
from unit_test_utils import *
from yolov6_obb.models.rbox_utils import non_max_suppression_rotation
import copy
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
logger.add("./logs/log_{time}.log", rotation="1 day")
def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--config', default='/home/crescent/YOLOv6-OBB/configs/yolov6m.py',help='Model configuration')
    parser.add_argument('--img_size', default=640,help = 'Dataloader img size')
    parser.add_argument('--anno_file_name', default="/home/crescent/YOLOv6-OBB/train_DroneVehicle.txt")
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument("--epochs", default=300, type=int, help="epochs")
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
def visual(img, bboxes):
    # [bs,n_anchors,[x,y,w,h,angle,conf,nc_class]]
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img* 255.0
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
    obb_boxes = bboxes[:,0:5]

    class_id = bboxes[:,-1:]

    bboxes = torch.cat([class_id,bboxes],-1)
    obb_vis(img,bboxes)
def label_assigner_loss_test(args):
    logger.info("The unit test of label assigner and loss calculation")
    #打出对应的log 计算
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else "cpu")
    cfg = Config.fromfile(args.config)
    # 半精度仅仅针对GPU设备进行处理
    model = build_model(cfg, num_classes=5, device=device)
    # VisDrone DataSet
    train_loader = create_dataloader(args.anno_file_name, args.img_size[0], args.batch_size,
                                     args.local_rank,args.workers,shuffle=False)[0]
    model.train() #切换到训练模型是
    compute_loss = ComputeLoss(num_classes=5,
                                        ori_img_size=args.img_size[0],
                                        use_reg_dfl=cfg.model.head.use_reg_dfl,
                                        reg_max= cfg.model.head.reg_max,
                                        use_angle_dfl=cfg.model.head.use_angle_dfl,
                                        angle_max= cfg.model.head.angle_max,
                                        iou_type=cfg.model.head.iou_type)
    pbar = enumerate(train_loader)
    optimizer = get_optimizer(args,cfg,model)
    scheduler, lf = get_lr_scheduler(args,cfg,optimizer)
    last_opt_step = -1
    scaler = amp.GradScaler(enabled=args.device != 'cpu')
    max_stepnum = len(train_loader)
    warmup_stepnum = max(round(cfg.solver.warmup_epochs * max_stepnum),1000)
    loss_num = 4
    data_info = {"car","freight car","truck","bus","van"}
    print(len(train_loader))
    for epoch_num in range(args.epochs):
        logger.info(f"The current epoch is{epoch_num}")
        if(epoch_num>=0):
            scheduler.step()
        optimizer.zero_grad()
        mean_loss = torch.zeros(loss_num, device=device)
        pbar = enumerate(train_loader)
        for step_num, batch_data in pbar:
            images, targets = prepro_data(batch_data, device)
            with amp.autocast(enabled=args.device != 'cpu'):
                preds, s_feature_maps = model(images)

                total_loss, loss_items = compute_loss(preds, targets, epoch_num, step_num)
            mean_loss = (mean_loss*step_num+loss_items)/(step_num+1)
            if(step_num%50==0):
                logger.info(('%10s' + '%10.4g' * loss_num) % (f'{epoch_num}/{args.epochs - 1}', \
                                                                *(mean_loss)))
            scaler.scale(total_loss).backward()
            curr_step = step_num + max_stepnum * epoch_num
            accumulate = max(1, round(64/args.batch_size))
            if curr_step <= warmup_stepnum:
                    accumulate = max(1, np.interp(curr_step, [0, warmup_stepnum], [1, 64 / args.batch_size]).round())
                    for k, param in enumerate(optimizer.param_groups):
                        warmup_bias_lr = cfg.solver.warmup_bias_lr if k == 2 else 0.0
                        param['lr'] = np.interp(curr_step, [0, warmup_stepnum], [warmup_bias_lr, param['initial_lr'] * lf(epoch_num)])
                        # logger.info(param['lr'])
                        if 'momentum' in param:
                            param['momentum'] = np.interp(curr_step, [0, warmup_stepnum], [cfg.solver.warmup_momentum, cfg.solver.momentum])
            if curr_step - last_opt_step >= accumulate:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    last_opt_step = curr_step
            last_opt_step = curr_step
        logger.info(f"{total_loss}")
        if(epoch_num>=30 and epoch_num%10==0):
            eval_nms(copy.deepcopy(model),args,cfg,data_info,train_loader)
        
@torch.no_grad()
def eval_nms(model,args,cfg, data_info, data_loader):
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else "cpu")
    half = True
    if half:
       model.half()
    model.eval()
    nc = 5
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    seen = 0
    single_cls = False  
    names = {0: data_info["item"][0]} if single_cls else {k: v for k, v in enumerate([i for i in data_info])}
    class_map = list(range(1000))
    
    p, r, f1, mp, mr, map50, map75, map, t0, t1, t2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    AP_method="voc07"
    save_dir=Path("")
    plots = False
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, batch_data in enumerate(data_loader):
        images, targets = prepro_data(batch_data, device)
        images = images.half()
        targets = targets.half()
        outputs, _ = model(images)
        outputs[...,4] = outputs[...,4]*180/np.pi
        # outputs [bs, n_anchors, 11]
        # [x,y,w,h,theta,conf,theta,cl1,cl2,....,cln] 
        # theta [0,90)
        
        eval_outputs = non_max_suppression_rotation(outputs,conf_thres=0.1,iou_thres=0.5)
        if(len(eval_outputs[0])!=0):
            demo = 0
            visual(images[demo:demo+1,:,:], eval_outputs[demo])
        # print(outputs[0].shape)
        # import copy
        # eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])
        # # 迁移到cpu中进行实现
        # visual(images, eval_outputs)
        # ipdb.set_trace()
        for si, pred in enumerate(eval_outputs):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else [] 
            seen += 1
            if (len(pred)==0):
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            predn = pred.clone()
            if nl:
                labelsn = labels.clone()
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))  # (correct, conf, pcls, tcls)
          
        # print('hello world')       
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, AP_method=AP_method, plot=plots, save_dir=save_dir, names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    s = ("%20s" + "%11s" * 7) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.75", "mAP@.5:.95")
    logger.info(s)
    pf = "%20s" + "%11i" * 2 + "%11.4g" * 5  # logger.info format
    logger.info(pf % ("All", seen, nt.sum(), mp, mr, map50, map75, map))      
    model.train()
    
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
    # return x if export_mode is True else [x, featmaps]
    x = model(img)
    
    logger.info("\n Starting to export onnx")
    export_file = '/home/crescent/YOLOv6-OBB/demo.onnx'
    with BytesIO() as f:
        torch.onnx.export(model, img, f, verbose=False, opset_version=13,
                          training = torch.onnx.TrainingMode.TRAINING,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['cls_score','reg_dist'],
                          )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model) 
        logger.info("The model has been successfully exported")
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, export_file)
    
if __name__ == "__main__":
    args = make_args()
    label_assigner_loss_test(args)
            