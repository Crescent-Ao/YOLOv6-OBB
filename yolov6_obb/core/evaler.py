#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
from pathlib import Path
from loguru import logger
import cv2 as cv2

from yolov6_obb.data.data_load import create_dataloader
from yolov6_obb.models.rbox_utils import non_max_suppression_rotation
from yolov6_obb.utils.checkpoint import load_checkpoint
from yolov6_obb.utils.torch_utils import time_sync, get_model_info

'''
python tools/eval.py --task 'train'/'val'/'speed'
'''


class Evaler:
    def __init__(self,
                 data,
                 batch_size=32,
                 img_size=640,
                 conf_thres=0.03,
                 iou_thres=0.65,
                 device='',
                 half=True,
                 save_dir='',
                 test_load_size=640,
                 verbose=False,
                 do_dota_metric=False,
                 do_pr_metric=True,
                 plot_curve=True,
                 plot_confusion_matrix=False
                 ):
        assert do_pr_metric or do_dota_metric, 'ERROR: at least set one val metric'
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.half = half
        self.save_dir = save_dir
        self.test_load_size = test_load_size
        self.verbose = verbose
        self.do_dota_metric = do_dota_metric
        self.do_pr_metric = do_pr_metric
        self.plot_curve = plot_curve
        self.plot_confusion_matrix = plot_confusion_matrix

    def init_model(self, model, weights, task):
        if task != 'train':
            model = load_checkpoint(weights, map_location=self.device)
            self.stride = int(model.stride.max())
            if self.device.type != 'cpu':
                model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))
            # switch to deploy
            from yolov6_obb.layers.common import RepVGGBlock
            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
            logger.info("Switch model to deploy modality.")
            logger.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        model.half() if self.half else model.float()
        return model

    def init_data(self, dataloader, task):
        '''Initialize dataloader.
        Returns a dataloader for task val or speed.
        '''
        """
            def create_dataloader(
    anno_file_name,
    img_size,
    batch_size,
    rank = -1,
    workers = 8,
    shuffle = True,
    data_dict = None,
    task = "train"
):
        """
        # dataloader = create_dataloader(self.data['val'],self.img_size,data_dict=self.data, task=task)[0]
        dataloader = create_dataloader(self.data['val'],self.img_size, self.batch_size,data_dict=self.data,task='val')[0]
        
        return dataloader

    def predict_model(self, model, dataloader, task):
        '''Model prediction
        Predicts the whole dataset and gets the prediced results and inference time.
        '''
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.")

        # whether to compute metric and plot PR curve and P、R、F1 curve under iou50 match rule
        if self.do_pr_metric:
            stats, ap = [], []
            seen = 0
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()
            if self.plot_confusion_matrix:
                from yolov6_obb.utils.metrics import ConfusionMatrix
                confusion_matrix = ConfusionMatrix(nc=model.nc)

        for i, (imgs, targets, paths) in enumerate(pbar):

            # pre-process
            t1 = time_sync()
            import ipdb
            imgs = imgs.to(self.device, non_blocking=True)
            imgs = imgs.half() if self.half else imgs.float()
            targets = targets.half() if self.half else targets.float().to(self.device)
            self.speed_result[1] += time_sync() - t1  # pre-process time

            # Inference
            t2 = time_sync()
            outputs, _ = model(imgs)
            self.speed_result[2] += time_sync() - t2  # inference time

            # post-process
            t3 = time_sync()
            outputs = non_max_suppression_rotation(outputs, self.conf_thres, self.iou_thres, multi_label=True)
            self.speed_result[3] += time_sync() - t3  # post-process time
            self.speed_result[0] += len(outputs)

            if self.do_pr_metric:
                import copy
                eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])
                
            # save result
            # pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))

            # for tensorboard visualization, maximum images to show: 8
            if i == 0:
                vis_num = min(len(imgs), 8)
                vis_outputs = outputs[:vis_num]
                vis_paths = paths[:vis_num]

            if not self.do_pr_metric:
                continue

            # Statistics per image
            # This code is based on
            # https://github.com/ultralytics/yolov5/blob/master/val.py
            for si, pred in enumerate(eval_outputs):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                if nl:
                    
                    labelsn = labels.clone()

                    from yolov6_obb.utils.metrics import process_batch

                    correct = process_batch(predn, labelsn.detach().cpu(), iouv)
                    if self.plot_confusion_matrix:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))  # (correct, conf, pcls, tcls)

        if self.do_pr_metric:
            # Compute statistics
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():

                from yolov6_obb.utils.metrics import ap_per_class
                p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plot_curve, save_dir=self.save_dir, names=model.names)
                AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() -1
                logger.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=model.nc)  # number of targets per class

                # Print results
                s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                logger.info(s)
                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
                logger.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

                self.pr_metric_result = (map50, map)

                # Print results per class
                if self.verbose and model.nc > 1:
                    for i, c in enumerate(ap_class):
                        logger.info(pf % (model.names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                           f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

                if self.plot_confusion_matrix:
                    confusion_matrix.plot(save_dir=self.save_dir, names=list(model.names))
            else:
                logger.info("Calculate metric failed, might check dataset.")
                self.pr_metric_result = (0.0, 0.0)

        return pred_results, vis_outputs, vis_paths


   

    def eval_speed(self, task):
        '''Evaluate model inference speed.'''
        if task != 'train':
            n_samples = self.speed_result[0].item()
            pre_time, inf_time, nms_time = 1000 * self.speed_result[1:].cpu().numpy() / n_samples
            for n, v in zip(["pre-process", "inference", "NMS"],[pre_time, inf_time, nms_time]):
                logger.info("Average {} time: {:.2f} ms".format(n, v))

    @staticmethod
    def check_task(task):
        if task not in ['train', 'val', 'test', 'speed']:
            raise Exception("task argument error: only support 'train' / 'val' / 'test' / 'speed' task.")

    @staticmethod
    def check_thres(conf_thres, iou_thres, task):
        '''Check whether confidence and iou threshold are best for task val/speed'''
        if task != 'train':
            if task == 'val' or task == 'test':
                if conf_thres > 0.03:
                    logger.warning(f'The best conf_thresh when evaluate the model is less than 0.03, while you set it to: {conf_thres}')
                if iou_thres != 0.65:
                    logger.warning(f'The best iou_thresh when evaluate the model is 0.65, while you set it to: {iou_thres}')
            if task == 'speed' and conf_thres < 0.4:
                logger.warning(f'The best conf_thresh when test the speed of the model is larger than 0.4, while you set it to: {conf_thres}')

    @staticmethod
    def reload_device(device, model, task):
        # device = 'cpu' or '0' or '0,1,2,3'
        if task == 'train':
            device = next(model.parameters()).device
        else:
            if device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            elif device:
                os.environ['CUDA_VISIBLE_DEVICES'] = device
                assert torch.cuda.is_available()
            cuda = device != 'cpu' and torch.cuda.is_available()
            device = torch.device('cuda:0' if cuda else 'cpu')
        return device


    def eval_model(self, pred_results, model, dataloader, task):
        '''Evaluate models
        For task speed, this function only evaluates the speed of model and outputs inference time.
        For task val, this function evaluates the speed and mAP by pycocotools, and returns
        inference time and mAP value.
        '''
        logger.info(f'\nEvaluating speed.')
        self.eval_speed(task)

        if not self.do_dota_metric and self.do_pr_metric:
            return self.pr_metric_result
        # logger.info(f'\nEvaluating mAP by pycocotools.')
        # if task != 'speed' and len(pred_results):
        #     if 'anno_path' in self.data:
        #         anno_json = self.data['anno_path']
        #     else:
        #        # generated coco format labels in dataset initialization
        #         task = 'val' if task == 'train' else task
        #         dataset_root = os.path.dirname(os.path.dirname(self.data[task]))
        #         base_name = os.path.basename(self.data[task])
        #         anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')
        #     pred_json = os.path.join(self.save_dir, "predictions.json")
        #     logger.info(f'Saving {pred_json}...')
        #     with open(pred_json, 'w') as f:
        #         json.dump(pred_results, f)

        #     anno = COCO(anno_json)
        #     pred = anno.loadRes(pred_json)
        #     cocoEval = COCOeval(anno, pred, 'bbox')
        #     if self.is_coco:
        #         imgIds = [int(os.path.basename(x).split(".")[0])
        #                     for x in dataloader.dataset.img_paths]
        #         cocoEval.params.imgIds = imgIds
        #     cocoEval.evaluate()
        #     cocoEval.accumulate()

        #     print each class ap from pycocotool result
        #     if self.verbose:

        #         import copy
        #         val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
        #         val_dataset_anns_count = 0
        #         label_count_dict = {"images":set(), "anns":0}
        #         label_count_dicts = [copy.deepcopy(label_count_dict) for _ in range(model.nc)]
        #         for _, ann_i in cocoEval.cocoGt.anns.items():
        #             if ann_i["ignore"]:
        #                 continue
        #             val_dataset_anns_count += 1
        #             nc_i = self.coco80_to_coco91_class().index(ann_i['category_id']) if self.is_coco else ann_i['category_id']
        #             label_count_dicts[nc_i]["images"].add(ann_i["image_id"])
        #             label_count_dicts[nc_i]["anns"] += 1

        #         s = ('%-16s' + '%12s' * 7) % ('Class', 'Labeled_images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
        #         LOGGER.info(s)
        #         IOU , all p, all cats, all gt, maxdet 100
        #         coco_p = cocoEval.eval['precision']
        #         coco_p_all = coco_p[:, :, :, 0, 2]
        #         map = np.mean(coco_p_all[coco_p_all>-1])

        #         coco_p_iou50 = coco_p[0, :, :, 0, 2]
        #         map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])
        #         mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
        #         mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        #         mf1 = 2 * mp * mr / (mp + mr + 1e-16)
        #         i = mf1.argmax()  # max F1 index

        #         pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
        #         LOGGER.info(pf % ('all', val_dataset_img_count, val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))

        #         compute each class best f1 and corresponding p and r
        #         for nc_i in range(model.nc):
        #             coco_p_c = coco_p[:, :, nc_i, 0, 2]
        #             map = np.mean(coco_p_c[coco_p_c>-1])

        #             coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
        #             map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])
        #             p = coco_p_c_iou50
        #             r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        #             f1 = 2 * p * r / (p + r + 1e-16)
        #             i = f1.argmax()
        #             LOGGER.info(pf % (model.names[nc_i], len(label_count_dicts[nc_i]["images"]), label_count_dicts[nc_i]["anns"], p[i], r[i], f1[i], map50, map))
        #     cocoEval.summarize()
        #     map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        #     Return results
        #     model.float()  # for training
        #     if task != 'train':
        #         LOGGER.info(f"Results saved to {self.save_dir}")
        #     return (map50, map)
        return (0.0, 0.0)


 