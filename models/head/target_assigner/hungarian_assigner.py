import torch
from scipy.optimize import linear_sum_assignment
from ops.iou3d_nms import iou3d_nms_cuda

def height_overlaps(boxes1, boxes2):

    boxes1_top_height = (boxes1[:,2]+ boxes1[:,5]).view(-1, 1)
    boxes1_bottom_height = boxes1[:,2].view(-1, 1)
    boxes2_top_height = (boxes2[:,2]+boxes2[:,5]).view(1, -1)
    boxes2_bottom_height = boxes2[:,2].view(1, -1)

    heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
    lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
    overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
    return overlaps_h

def overlaps(boxes1, boxes2):

    rows = len(boxes1)
    cols = len(boxes2)
    if rows * cols == 0:
        return boxes1.new(rows, cols)

    overlaps_h = height_overlaps(boxes1, boxes2)
    boxes1_bev = boxes1[:,:7]
    boxes2_bev = boxes2[:,:7]

    overlaps_bev = boxes1_bev.new_zeros(
        (boxes1_bev.shape[0], boxes2_bev.shape[0])
    ).cuda()
    iou3d_nms_cuda.boxes_overlap_bev_gpu(
        boxes1_bev.contiguous().cuda(), boxes2_bev.contiguous().cuda(), overlaps_bev
    )

    overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

    volume1 = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]).view(-1, 1)
    volume2 = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)

    return iou3d

class HungarianAssigner3D:
    def __init__(self, cls_cost, reg_cost, iou_cost):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost

    def focal_loss_cost(self, cls_pred, gt_labels):
        weight = self.cls_cost.get('weight', 0.15)
        alpha = self.cls_cost.get('alpha', 0.25)
        gamma = self.cls_cost.get('gamma', 2.0)
        eps = self.cls_cost.get('eps', 1e-12)

        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (
            1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (
            1 - cls_pred).pow(gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * weight

    def bevbox_cost(self, bboxes, gt_bboxes, point_cloud_range):
        weight = self.reg_cost.get('weight', 0.25)

        pc_start = bboxes.new(point_cloud_range[0:2])
        pc_range = bboxes.new(point_cloud_range[3:5]) - bboxes.new(point_cloud_range[0:2])

        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * weight

    def iou3d_cost(self, bboxes, gt_bboxes):
        iou = overlaps(bboxes, gt_bboxes)
        weight = self.iou_cost.get('weight', 0.25)
        iou_cost = - iou
        return iou_cost * weight, iou

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, point_cloud_range):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:

            if num_gts == 0:

                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, max_overlaps, assigned_labels

        cls_cost = self.focal_loss_cost(cls_pred[0].T, gt_labels)
        reg_cost = self.bevbox_cost(bboxes, gt_bboxes, point_cloud_range)
        iou_cost, iou = self.iou3d_cost(bboxes, gt_bboxes)

        cost = cls_cost + reg_cost + iou_cost

        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        assigned_gt_inds[:] = 0

        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]

        return assigned_gt_inds, max_overlaps
