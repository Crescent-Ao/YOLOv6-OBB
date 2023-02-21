import torch
import torch.nn as nn  
import torch.nn.functional as F  
import mmcv.ops.box_iou_rotated as box_iou_rotated
import yolov6_obb.utils.obb_utils.rotated_iou_similarity
class RotatedTaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(RotatedTaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, num_anchor_list, gt_labels,gt_bboxes, gt_mask, bg_index,gt_scores=None):
        """
        The assignment is done in following steps
            1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
            2. select top-k bbox as candidates for each gt
            3. limit the positive sample's center in gt (because the anchor-free detector
                only can predict positive distance)
            4. if an anchor box is assigned to multiple gts, the one with the
            
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 5)
            anchor_points (Tensor, float32): pre-defined anchors, shape(1, L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 5)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 5)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3
        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 5])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores 
        ious = torch.where(ious > 1 + self.eps, torch.zeros_like(ious), ious)
        pred_scores = pred_scores.permyte(0,2,1)
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        gt_labels_ind = torch.stack(
            [batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)],
            axis=-1)
        
        
def g