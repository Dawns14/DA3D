import torch
try:
    from cuda_op.cuda_ext import sort_v
except:
    from utils.Rotated_IoU.cuda_op.cuda_ext import sort_v

EPSILON = 1e-8

def box_intersection_th(corners1:torch.Tensor, corners2:torch.Tensor):

    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)

    line1_ext = line1.unsqueeze(3).repeat([1,1,1,4,1])
    line2_ext = line2.unsqueeze(2).repeat([1,1,4,1,1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]

    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)
    mask = mask_t * mask_u
    t = den_t / (num + EPSILON)
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask

def box1_in_box2(corners1:torch.Tensor, corners2:torch.Tensor):

    a = corners2[:, :, 0:1, :]
    b = corners2[:, :, 1:2, :]
    d = corners2[:, :, 3:4, :]
    ab = b - a
    am = corners1 - a
    ad = d - a
    p_ab = torch.sum(ab * am, dim=-1)
    norm_ab = torch.sum(ab * ab, dim=-1)
    p_ad = torch.sum(ad * am, dim=-1)
    norm_ad = torch.sum(ad * ad, dim=-1)

    cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)
    cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)
    return cond1*cond2

def box_in_box_th(corners1:torch.Tensor, corners2:torch.Tensor):

    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1

def build_vertices(corners1:torch.Tensor, corners2:torch.Tensor,
                c1_in_2:torch.Tensor, c2_in_1:torch.Tensor,
                inters:torch.Tensor, mask_inter:torch.Tensor):

    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([B, N, -1, 2])], dim=2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2)
    return vertices, mask

def sort_indices(vertices:torch.Tensor, mask:torch.Tensor):

    num_valid = torch.sum(mask.int(), dim=2).int()
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean
    return sort_v(vertices_normalized, mask, num_valid).long()

def calculate_area(idx_sorted:torch.Tensor, vertices:torch.Tensor):

    idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected

def oriented_box_intersection_2d(corners1:torch.Tensor, corners2:torch.Tensor):

    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)
