import torch
import torch.nn as nn
import numpy as np
import nms

from utils.Rotated_IoU.oriented_iou_loss import cal_iou

class RdrSpcubeHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        if self.cfg_dataset_ver2:
            roi = self.cfg.DATASET.roi
            x_min, y_min, z_min, x_max, y_max, z_max = roi.xyz
            self.roi = {
                'x': [x_min, x_max],
                'y': [y_min, y_max],
                'z': [z_min, z_max],
            }
            self.grid_size = roi.grid_size
        else:
            self.roi = self.cfg.DATASET.RDR_SP_CUBE.ROI
            self.grid_size = self.cfg.DATASET.RDR_SP_CUBE.GRID_SIZE
        try:
            self.nms_thr = self.cfg.MODEL.HEAD.NMS_OVERLAP_THRESHOLD
        except:
            print('* Exception error (Head): nms threshold is set as 0.3')
            self.nms_thr = 0.3

        self.anchor_per_grid = []
        num_anchor_temp = 0

        self.list_anchor_classes = []
        self.list_anchor_matched_thr = []
        self.list_anchor_unmatched_thr = []
        self.list_anchor_targ_idx = []
        self.list_anchor_idx = []

        self.list_anc_idx_to_cls_id = []
        if self.cfg_dataset_ver2:
            dict_label = self.cfg.DATASET.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            self.dict_cls_name_to_id = dict()
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                self.dict_cls_name_to_id[k] = logit_idx
                self.dict_cls_name_to_id['Background'] = 0
        else:
            self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID

        num_prior_anchor_idx = 0
        for info_anchor in self.cfg.MODEL.ANCHOR_GENERATOR_CONFIG:
            now_cls_name = info_anchor['class_name']
            self.list_anchor_classes.append(now_cls_name)
            self.list_anchor_matched_thr.append(info_anchor['matched_threshold'])
            self.list_anchor_unmatched_thr.append(info_anchor['unmatched_threshold'])

            self.anchor_sizes = info_anchor['anchor_sizes']
            self.anchor_rotations = info_anchor['anchor_rotations']
            self.anchor_bottoms = info_anchor['anchor_bottom_heights']

            self.list_anchor_targ_idx.append(num_prior_anchor_idx)
            num_now_anchor = int(len(self.anchor_sizes)*len(self.anchor_rotations)*len(self.anchor_bottoms))
            num_now_anchor_idx = num_prior_anchor_idx+num_now_anchor
            self.list_anchor_idx.append(num_now_anchor_idx)
            num_prior_anchor_idx = num_now_anchor_idx

            for anchor_size in self.anchor_sizes:
                for anchor_rot in self.anchor_rotations:
                    for anchor_bottom in self.anchor_bottoms:
                        temp_anchor = [anchor_bottom] + anchor_size + [np.cos(anchor_rot), np.sin(anchor_rot)]
                        num_anchor_temp += 1
                        self.anchor_per_grid.append(temp_anchor)
                        self.list_anc_idx_to_cls_id.append(self.dict_cls_name_to_id[now_cls_name])
        self.num_anchor_per_grid = num_anchor_temp

        self.num_box_code = len(self.cfg.MODEL.HEAD.BOX_CODE)

        input_channels = self.cfg.MODEL.HEAD.DIM
        self.conv_cls = nn.Conv2d(
            input_channels, 1 + self.num_anchor_per_grid,
            kernel_size=1
        )
        self.conv_reg = nn.Conv2d(
            input_channels, self.num_anchor_per_grid*self.num_box_code,
            kernel_size=1
        )

        self.bg_weight = cfg.MODEL.HEAD.BG_WEIGHT
        self.categorical_focal_loss = FocalLoss()
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING

        self.anchor_map_for_batch = self.create_anchors().cuda()

    def forward(self, dict_item):
        spatial_features_2d = dict_item['bev_feat']

        cls_pred = self.conv_cls(spatial_features_2d)
        reg_pred = self.conv_reg(spatial_features_2d)

        dict_item['pred'] = {
            'cls': cls_pred,
            'reg': reg_pred,
        }

        return dict_item

    def create_anchors(self):

        dtype = torch.float32
        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        grid_size = self.grid_size
        n_x = int((x_max-x_min)/grid_size)
        n_y = int((y_max-y_min)/grid_size)

        half_grid_size = grid_size/2.
        anchor_y = torch.arange(y_min, y_max, grid_size, dtype=dtype) - half_grid_size
        anchor_x = torch.arange(x_min, x_max, grid_size, dtype=dtype) - half_grid_size

        anchor_y = anchor_y.repeat_interleave(n_x)
        anchor_x = anchor_x.repeat(n_y)

        flattened_anchor_map = torch.stack((anchor_x, anchor_y), dim=1).unsqueeze(0).repeat(self.num_anchor_per_grid, 1, 1)

        flattened_anchor_attr = torch.tensor(self.anchor_per_grid, dtype=dtype)

        flattened_anchor_attr = flattened_anchor_attr.unsqueeze(1).repeat(1, flattened_anchor_map.shape[1], 1)

        anchor_map = torch.cat((flattened_anchor_map, flattened_anchor_attr),\
            dim=-1).view(self.num_anchor_per_grid, n_y, n_x, 8).contiguous().permute(0,3,1,2)
        anchor_map = anchor_map.reshape(-1, n_y, n_x).contiguous()

        anchor_map_for_batch = anchor_map.unsqueeze(0)

        return anchor_map_for_batch

    def loss(self, dict_item):
        cls_pred = dict_item['pred']['cls']
        reg_pred = dict_item['pred']['reg']

        dtype, device = cls_pred.dtype, cls_pred.device
        B, _, n_y, n_x = cls_pred.shape
        num_grid_per_anchor = int(n_y*n_x)

        anchor_maps = self.anchor_map_for_batch.repeat(B, 1, 1, 1)

        reg_pred = anchor_maps + reg_pred

        cls_pred = cls_pred.view(B, 1+self.num_anchor_per_grid, n_y, n_x)
        reg_pred = reg_pred.view(B, self.num_anchor_per_grid, -1, n_y, n_x)

        anc_idx_targets = torch.full((B, n_y, n_x), -1, dtype = torch.long, device = device)

        pos_reg_pred = []
        pos_reg_targ = []

        is_label_contain_objs = False
        for batch_idx, list_objs in enumerate(dict_item['label']):
            if len(list_objs) != 0:
                is_label_contain_objs = True

            prior_anc_idx = 0
            list_anchor_per_cls = []
            for idx_anc_cls, anc_cls_name in enumerate(self.list_anchor_classes):
                now_anc_idx = self.list_anchor_idx[idx_anc_cls]

                temp_anc = torch.cat(\
                        (reg_pred[batch_idx,prior_anc_idx:now_anc_idx,:2],\
                         reg_pred[batch_idx,prior_anc_idx:now_anc_idx,3:5],\
                         torch.atan(reg_pred[batch_idx,prior_anc_idx:now_anc_idx,6:7]/reg_pred[batch_idx,prior_anc_idx:now_anc_idx,5:6])), dim=1)

                temp_anc = temp_anc.permute(0, 2, 3, 1).contiguous()
                temp_anc = temp_anc.view(1,-1,5)

                list_anchor_per_cls.append(temp_anc)
                prior_anc_idx = now_anc_idx

            for label_idx, label in enumerate(list_objs):
                cls_name, cls_id, (xc, yc, zc, rz, xl, yl, zl), _ = label

                idx_anc_cls = self.list_anchor_classes.index(cls_name)
                pred_anchors = list_anchor_per_cls[idx_anc_cls]
                cls_targ_idx = self.list_anchor_targ_idx[idx_anc_cls]
                matched_iou_thr = self.list_anchor_matched_thr[idx_anc_cls]
                unmatched_iou_thr = self.list_anchor_unmatched_thr[idx_anc_cls]

                label_anchor = torch.tensor([xc, yc, xl, yl, rz], dtype=dtype, device=device)

                label_anchor = label_anchor.unsqueeze(0).unsqueeze(0).repeat(1, pred_anchors.shape[1], 1)

                iou, _, _, _ = cal_iou(label_anchor, pred_anchors)

                pos_iou_anc_idx = torch.where(iou > matched_iou_thr)[1]

                if len(pos_iou_anc_idx) == 0:

                    pos_iou_anc_idx = (torch.argmax(iou)).reshape(1)

                neg_iou_anc_idx = torch.where(iou < unmatched_iou_thr)[1]

                neg_iou_anc_idx = torch.remainder(neg_iou_anc_idx, num_grid_per_anchor)

                idx_y_neg = torch.div(neg_iou_anc_idx, n_x, rounding_mode='trunc')
                idx_x_neg = torch.remainder(neg_iou_anc_idx, n_x)

                anc_idx_targets[batch_idx, idx_y_neg, idx_x_neg] = 0

                pos_iou_anc_idx_offset = torch.div(pos_iou_anc_idx,\
                    num_grid_per_anchor, rounding_mode='trunc')
                pos_iou_anc_idx = torch.remainder(pos_iou_anc_idx, num_grid_per_anchor)

                idx_y_pos = torch.div(pos_iou_anc_idx, n_x, rounding_mode='trunc')
                idx_x_pos = torch.remainder(pos_iou_anc_idx, n_x)

                temp_anc_idx_targets = cls_targ_idx + pos_iou_anc_idx_offset
                anc_idx_targets[batch_idx, idx_y_pos, idx_x_pos] = 1 + temp_anc_idx_targets

                temp_reg_box_pred = reg_pred[batch_idx,temp_anc_idx_targets,:,idx_y_pos,idx_x_pos]
                temp_num_pos, _ = temp_reg_box_pred.shape
                temp_reg_box_targ = torch.tensor([[xc, yc, zc, xl, yl, zl,\
                    np.cos(rz), np.sin(rz)]], dtype = dtype, device = device).repeat((temp_num_pos, 1))

                pos_reg_pred.append(temp_reg_box_pred)
                pos_reg_targ.append(temp_reg_box_targ)

        if not is_label_contain_objs:
            loss_reg = 0.
            loss_cls = 0.
        else:

            counted_anc_idx = torch.where(anc_idx_targets > -1)

            anc_idx_targets_counted = anc_idx_targets[counted_anc_idx]

            anc_logit_counted = cls_pred[counted_anc_idx[0],:,counted_anc_idx[1],counted_anc_idx[2]]

            anc_cls_weights = torch.ones(1+self.num_anchor_per_grid, device = device)
            for idx_anc in range(1+self.num_anchor_per_grid):
                len_targ_anc = float(len(torch.where(anc_idx_targets_counted==idx_anc)[0]))
                if idx_anc == 0:
                    temp_weight = self.bg_weight/len_targ_anc
                else:
                    if len_targ_anc == 0.:
                        temp_weight = 0.
                    else:
                        temp_weight = 1./len_targ_anc
                anc_cls_weights[idx_anc] = min(temp_weight, 1.)

            self.categorical_focal_loss.weight = anc_cls_weights
            loss_cls = self.categorical_focal_loss(anc_logit_counted, anc_idx_targets_counted)

            pos_reg_pred = torch.cat(pos_reg_pred, dim=0)
            pos_reg_targ = torch.cat(pos_reg_targ, dim=0)
            loss_reg = torch.nn.functional.smooth_l1_loss(pos_reg_pred, pos_reg_targ)
        total_loss = loss_cls + loss_reg

        if self.is_logging:
            dict_item['logging'] = dict()
            dict_item['logging'].update(self.logging_dict_loss(total_loss, 'total_loss'))
            dict_item['logging'].update(self.logging_dict_loss(loss_reg, 'loss_reg'))
            dict_item['logging'].update(self.logging_dict_loss(loss_cls, 'focal_loss_cls'))

        return total_loss

    def logging_dict_loss(self, loss, name_key):
        try:
            log_loss = loss.cpu().detach().item()
        except:
            log_loss = loss

        return {name_key: log_loss}

    def get_nms_pred_boxes_for_single_sample(self, dict_item, conf_thr, is_nms=True):

        cls_pred = dict_item['pred']['cls'][0]
        reg_pred = dict_item['pred']['reg'][0]
        anchor_map = self.anchor_map_for_batch[0]
        reg_pred = anchor_map + reg_pred

        device = cls_pred.device

        cls_pred = cls_pred.view(cls_pred.shape[0], -1)
        reg_pred = reg_pred.view(reg_pred.shape[0], -1)

        cls_pred = torch.softmax(cls_pred, dim=0)
        idx_deal = torch.where(
            (torch.argmax(cls_pred, dim=0)!=0) & (torch.max(cls_pred, dim=0)[0]>conf_thr))

        tensor_anc_idx_per_cls = torch.tensor(self.list_anc_idx_to_cls_id, dtype=torch.long, device=device)

        len_deal_anc = len(idx_deal[0])

        if len_deal_anc > 0:
            grid_anc_cls_logit = cls_pred[:, idx_deal[0]]
            grid_anc_cls_idx = torch.argmax(grid_anc_cls_logit, dim=0)
            grid_reg = reg_pred[:, idx_deal[0]]

            idx_range_anc = torch.arange(0, len_deal_anc, dtype=torch.long, device=device)
            anc_conf_score = grid_anc_cls_logit[grid_anc_cls_idx,idx_range_anc].unsqueeze(0)
            grid_anc_cls_idx = grid_anc_cls_idx -1

            list_sliced_reg_bbox = []
            idx_slice_start = (grid_anc_cls_idx*self.num_box_code).long()

            for idx_reg_value in range(self.num_box_code):
                list_sliced_reg_bbox.append(grid_reg[idx_slice_start+idx_reg_value,idx_range_anc])
            sliced_reg_bbox = torch.stack(list_sliced_reg_bbox)

            temp_angle = torch.atan2(sliced_reg_bbox[-1,:], sliced_reg_bbox[-2,:]).unsqueeze(0)

            pred_reg_bbox_with_conf = torch.cat((anc_conf_score, sliced_reg_bbox[:-2,:], temp_angle), dim=0)
            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.transpose(0,1)

            cls_id_per_anc = tensor_anc_idx_per_cls[grid_anc_cls_idx]

            num_of_bbox = len_deal_anc

            try:
                if is_nms:
                    pred_reg_xy_xlyl_th = torch.cat((pred_reg_bbox_with_conf[:,1:3],\
                    pred_reg_bbox_with_conf[:,4:6], pred_reg_bbox_with_conf[:,7:8]), dim=1).cpu().detach().numpy()

                    c_list = list(map(tuple, pred_reg_xy_xlyl_th[:,:2]))

                    dim_list = list(map(np.abs, pred_reg_xy_xlyl_th[:,2:4]))

                    dim_list = list(map(tuple, pred_reg_xy_xlyl_th[:,2:4]))
                    angle_list = list(map(float, pred_reg_xy_xlyl_th[:,4]))

                    list_tuple_for_nms = [[a, b, c] for (a, b, c) in zip(c_list, dim_list, angle_list)]
                    conf_score = pred_reg_bbox_with_conf[:, 0:1].cpu().detach().numpy()

                    indices = nms.rboxes(list_tuple_for_nms, conf_score, nms_threshold=self.nms_thr)
                    pred_reg_bbox_with_conf = pred_reg_bbox_with_conf[indices]
                    cls_id_per_anc = cls_id_per_anc[indices]

                    num_of_bbox = len(indices)
            except:
                print('* Exception error (head.py): nms error, probably assert height > 0')

            pred_reg_bbox_with_conf = pred_reg_bbox_with_conf.cpu().detach().numpy().tolist()
            cls_id_per_anc = cls_id_per_anc.cpu().detach().numpy().tolist()

        else:

            pred_reg_bbox_with_conf = None
            cls_id_per_anc = None
            num_of_bbox = 0

        dict_item['pp_bbox'] = pred_reg_bbox_with_conf
        dict_item['pp_cls'] = cls_id_per_anc
        dict_item['pp_desc'] = dict_item['meta'][0]['desc']
        dict_item['pp_num_bbox'] = num_of_bbox

        return dict_item

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)

        return nn.functional.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )
