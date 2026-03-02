import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List

class NeighborVoxelSAModuleMSG(nn.Module):

    def __init__(self, *, query_ranges: List[List[int]], radii: List[float],
        nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method='max_pool'):

        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i]

            cur_mlp_in = nn.Sequential(
                nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[1])
            )

            cur_mlp_pos = nn.Sequential(
                nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp_spec[1])
            )

            cur_mlp_out = nn.Sequential(
                nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[2]),
                nn.ReLU()
            )

            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,\
                                        new_coords, features, voxel2point_indices):

        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        new_features_list = []
        for k in range(len(self.groupers)):

            features_in = features.permute(1, 0).unsqueeze(0)
            features_in = self.mlps_in[k](features_in)

            features_in = features_in.permute(0, 2, 1).contiguous()

            features_in = features_in.view(-1, features_in.shape[-1])

            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](
                new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices
            )
            grouped_features[empty_ball_mask] = 0

            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)

            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0

            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)

            position_features = self.mlps_pos[k](grouped_xyz)
            new_features = grouped_features + position_features
            new_features = self.relu(new_features)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)
            else:
                raise NotImplementedError

            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)
        return new_features
