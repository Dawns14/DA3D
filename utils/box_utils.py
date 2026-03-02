import numpy as np
import scipy
import torch
import copy
from scipy.spatial import Delaunay

from ops.roiaware_pool3d import roiaware_pool3d_utils
from . import common_utils

def in_hull(p, hull):

    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag

def boxes_to_corners_3d(boxes3d):

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def corners_rect_to_camera(corners):

    height_group = [(0, 4), (1, 5), (2, 6), (3, 7)]
    width_group = [(0, 1), (2, 3), (4, 5), (6, 7)]
    length_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    vector_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    height, width, length = 0., 0., 0.
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(height_group, width_group, length_group, vector_group):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])
        vector[0] += (corners[index_v[0], :] - corners[index_v[1], :])[0]
        vector[1] += (corners[index_v[0], :] - corners[index_v[1], :])[2]

    height, width, length = height*1.0/4, width*1.0/4, length*1.0/4
    rotation_y = -np.arctan2(vector[1], vector[0])

    center_point = corners.mean(axis=0)
    center_point[1] += height/2
    camera_rect = np.concatenate([center_point, np.array([length, height, width, rotation_y])])

    return camera_rect

def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1, use_center_to_filter=True):

    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    if use_center_to_filter:
        box_centers = boxes[:, 0:3]
        mask = ((box_centers >= limit_range[0:3]) & (box_centers <= limit_range[3:6])).all(axis=-1)
    else:
        corners = boxes_to_corners_3d(boxes)
        corners = corners[:, :, 0:2]
        mask = ((corners >= limit_range[0:2]) & (corners <= limit_range[3:5])).all(axis=2)
        mask = mask.sum(axis=1) >= min_num_corners

    return mask

def remove_points_in_boxes3d(points, boxes3d):

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):

    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):

    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    w, l, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] += h[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

def boxes3d_kitti_lidar_to_fakelidar(boxes3d_lidar):

    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    dx, dy, dz = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    heading = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] -= dz[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], dy, dx, dz, -heading - np.pi / 2], axis=-1)

def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d

def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):

    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)

    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):

    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])
    R_list = np.transpose(rot_list, (2, 0, 1))

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)
    rotated_corners = np.matmul(temp_corners, R_list)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):

    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)
    max_uv = np.max(corners_in_image, axis=1)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image

def boxes_iou_normal(boxes_a, boxes_b):

    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou

def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):

    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes

def boxes3d_nearest_bev_iou(boxes_a, boxes_b):

    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)

def area(box) -> torch.Tensor:

    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area

def pairwise_iou(boxes1, boxes2) -> torch.Tensor:

    area1 = area(boxes1)
    area2 = area(boxes2)

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )

    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    del width_height

    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def center_to_corner2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], device=dim.device).type_as(center)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners

def bbox3d_overlaps_diou(pred_boxes, gt_boxes):

    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) -\
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) -\
              torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious
