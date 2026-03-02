import numpy as np
import torch

def generate_table():

    skip = [[0,2], [1,3], [5,7], [4,6]]
    line = []
    points = []

    def all_except_two(o1, o2):
        a = []
        for i in range(8):
            if i != o1 and i != o2:
                a.append(i)
        return a

    for i in range(8):
        for j in range(i+1, 8):
            if [i, j] not in skip:
                line.append([i, j])
                points.append(all_except_two(i, j))
    return line, points

LINES, POINTS = generate_table()
LINES = np.array(LINES).astype(np.int)
POINTS = np.array(POINTS).astype(np.int)

def gather_lines_points(corners:torch.Tensor):

    dim = corners.dim()
    idx_lines = torch.LongTensor(LINES).to(corners.device).unsqueeze(-1)
    idx_points = torch.LongTensor(POINTS).to(corners.device).unsqueeze(-1)
    idx_lines = idx_lines.repeat(1,1,2)
    idx_points = idx_points.repeat(1,1,2)
    if dim > 2:
        for _ in range(dim-2):
            idx_lines = torch.unsqueeze(idx_lines, 0)
            idx_points = torch.unsqueeze(idx_points, 0)
        idx_points = idx_points.repeat(*corners.size()[:-2], 1, 1, 1)
        idx_lines = idx_lines.repeat(*corners.size()[:-2], 1, 1, 1)
    corners_ext = corners.unsqueeze(-3).repeat( *([1]*(dim-2)), 24, 1, 1)
    lines = torch.gather(corners_ext, dim=-2, index=idx_lines)
    points = torch.gather(corners_ext, dim=-2, index=idx_points)

    return lines, points, idx_lines, idx_points

def point_line_distance_range(lines:torch.Tensor, points:torch.Tensor):

    x1 = lines[..., 0:1, 0]
    y1 = lines[..., 0:1, 1]
    x2 = lines[..., 1:2, 0]
    y2 = lines[..., 1:2, 1]
    x = points[..., 0]
    y = points[..., 1]
    den = (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1

    num = torch.sqrt( (y2-y1).square() + (x2-x1).square() + 1e-14 )
    d = den/num
    d_max = d.max(dim=-1)[0]
    d_min = d.min(dim=-1)[0]
    d1 = d_max - d_min
    d2 = torch.max(d.abs(), dim=-1)[0]

    return torch.max(d1, d2)

def point_line_projection_range(lines:torch.Tensor, points:torch.Tensor):

    x1 = lines[..., 0:1, 0]
    y1 = lines[..., 0:1, 1]
    x2 = lines[..., 1:2, 0]
    y2 = lines[..., 1:2, 1]
    k = (y2 - y1)/(x2 - x1 + 1e-8)
    vec = torch.cat([torch.ones_like(k, dtype=k.dtype, device=k.device), k], dim=-1)
    vec = vec.unsqueeze(-2)
    points_ext = torch.cat([lines, points], dim=-2)
    den = torch.sum(points_ext * vec, dim=-1)
    proj = den / torch.norm(vec, dim=-1, keepdim=False)
    proj_max = proj.max(dim=-1)[0]
    proj_min = proj.min(dim=-1)[0]
    return proj_max - proj_min

def smallest_bounding_box(corners:torch.Tensor, verbose=False):

    lines, points, _, _ = gather_lines_points(corners)
    proj = point_line_projection_range(lines, points)
    dist = point_line_distance_range(lines, points)
    area = proj * dist

    zero_mask = (area == 0).type(corners.dtype)
    fake = torch.ones_like(zero_mask, dtype=corners.dtype, device=corners.device)* 1e8 * zero_mask
    area += fake
    area_min, idx = torch.min(area, dim=-1, keepdim=True)
    w = torch.gather(proj, dim=-1, index=idx)
    h = torch.gather(dist, dim=-1, index=idx)
    w = w.squeeze(-1).float()
    h = h.squeeze(-1).float()
    area_min = area_min.squeeze(-1).float()
    if verbose:
        return w, h, area_min, idx.squeeze(-1)
    else:
        return w, h

if __name__ == "__main__":

    from utiles import box2corners
    import matplotlib.pyplot as plt
    box1 = [0, 0, 2, 3, np.pi/6]
    box2 = [1, 5, 4, 4, -np.pi/4]
    corners1 = box2corners(*box1)
    corners2 = box2corners(*box2)
    tensor1 = torch.FloatTensor(np.concatenate([corners1, corners2], axis=0))
    w, h, a, i = smallest_bounding_box(tensor1, True)
    print("width:", w.item(), ". length:", h.item())
    print("area: ", a.item())
    print("index in 26 candidates: ", i.item())
    print("colliniear with points: ", LINES[i.item()])
    plt.scatter(corners1[:, 0], corners1[:, 1])
    plt.scatter(corners2[:, 0], corners2[:, 1])
    for i in range(corners1.shape[0]):
        plt.text(corners1[i, 0], corners1[i, 1], str(i))
    for i in range(corners2.shape[0]):
        plt.text(corners2[i, 0], corners2[i, 1], str(i+4))
    plt.axis("equal")
    plt.show()
