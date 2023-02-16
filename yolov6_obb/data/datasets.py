import os
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from loguru import logger
import tqdm
import time
import .data_augment as DataAug

class TrainValDataset(Dataset):
    def __init__(
        self,
        anno_file_name,
        stride=32,
        img_size=640,
        rank=-1,
        data_dict=None
    ):
        """
            YOLOv6-OBB Dataset
        Args:
            anno_file_name (str): The absolute annotation filaname
            stride (int, optional): The max stride. Defaults to 32.
            img_size (int, optional): The train img size. Defaults to 640.
            rank (int, optional): DDP. Defaults to -1.
            data_dict (dict, optional): The dict describes {"class_name":num}
        """
        t1 = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1,0)
        self.class_names = data_dict.keys()
        self.annotations = self.load_annotations(anno_file_name)
        t2 = time.time()
        
        if self.main_process:
            logger.info(f"%.1fs for dataset initialization"%(t2-t1))
    def __getitem__(self, item):
        img_org, bboxes_org = self.parse_annotation(item)
        img_org = img_org.transpose(2,0,1)
        item_mix = random.randint(0, len(self.annotations) - 1)
        img_mix, bboxes_mix = self.parse_annotation(item_mix)
        img_mix = img_mix.transpose(2, 0, 1)
        img, bboxes = DataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix   
        bboxes = self.create_label(bboxes)
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img,bboxes
        
    def __len__(self):
        return len(self.annotations)  
    def load_annotations(self,anno_file_name):
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x:len(x)>0,f.readlines()))
        assert len(annotations>0), f"There is no annotation in {self.anno_file_name}"
        annotations = [x.strip().split(' ') for x in annotations]
        return annotations
    def parser_annotation(self,index):
        cur_anno = self.annotations[index]
        img =  cv2.imread(cur_anno[0])
        assert img is not None, f"{cur_anno[0]} file not found"
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        img, bboxes = DataAug.RandomVerticalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.HSV()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        return img, bboxes
    def create_label(self,bboxes):
        # [none,class_id,xmin,ymin,xmax,ymax,c_x_r,c_y_r,a1,a2,a3,a4,area_ratio,angle]
        bboxes_out = torch.zeros(len(bboxes),2+4+2+4+2)
        for gt_label in bboxes:
            bbox_xyxy = gt_label[:4]
            bbox_obb = gt_label[5:13]
            xmin, ymin, xmax, ymax = bbox_xyxy
            bboxes_out[:,2:6] = bbox_xyxy
            c_x = (xmax + xmin) / 2
            c_y = (ymax + ymin) / 2
            if gt_label[13]>0.9:
                a1 = a2 = a3 = a4 = 0   
            else:
                a1 = (bbox_obb[0] - bbox_xyxy[0]) / box_w
                a2 = (bbox_obb[3] - bbox_xyxy[1]) / box_h
                a3 = (bbox_xyxy[2] - bbox_obb[4]) / box_w
                a4 = (bbox_xyxy[3] - bbox_obb[7]) / box_h
            class_id = int(gt_label[4])
            bboxes_out[:,1] = class_id
            c_x_r = (bbox_obb[0] + bbox_obb[2] + bbox_obb[4] + bbox_obb[6]) / 4
            c_y_r = (bbox_obb[1] + bbox_obb[3] + bbox_obb[5] + bbox_obb[7]) / 4
            # 样本的中心点，按照点来分配利用这个来分配
            # [none,class_id,xmin,ymin,xmax,ymax,c_x_r,c_y_r,a1,a2,a3,a4,area_ratio,angle]
            # 后续可以转换成角度
            angle = gt_label[14]*np.pi/180
            bboxes_out[:,6] = c_x_r
            bboxes_out[:,7] = c_y_r
            bboxes_out[:,8:12] = [a1,a2,a3,a4]
            bboxes_out[:,12] = gt_label[13]
            bboxes_out[:,13] = angle
        return bboxes_out          
    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        for i,l in enumerate(label):
            i[:,0] = i
        return torch.stack(img,0), torch.cat(label,0)        
            
        