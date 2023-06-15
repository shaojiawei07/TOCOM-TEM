import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from multiview_detector.models.resnet import resnet18
from compressai.entropy_models import EntropyBottleneck, GaussianConditional 
from multiview_detector.models.GaussianProbModel import GaussianLikelihoodEstimation
import matplotlib.pyplot as plt
import math
import copy



class PerspTransDetector(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.args = args
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

        base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
        split = 7
        self.base_pt1 = base[:split].to('cuda:1')
        self.base_pt2 = base[split:].to('cuda:0')

        channel = 8 
        half_channel = int(channel/2)

        self.feature_extraction = nn.Conv2d(512, channel, 1).to("cuda:0")



        self.hyperEncoder = nn.Sequential(
                                    nn.Conv2d(channel, half_channel, kernel_size = 3, stride = 1, padding= 1),
                                    nn.ReLU(),
                                    nn.Conv2d(half_channel, half_channel, kernel_size = 5, stride = 2, padding= 2),
                                    nn.ReLU(),
                                    nn.Conv2d(half_channel, half_channel, kernel_size = (3,4), stride = (3,4), padding= 0),
                                    nn.ReLU(),
                                    nn.Conv2d(half_channel, half_channel, kernel_size = (3,4), stride = (3,4), padding= 0),
                                    nn.ReLU(),
                                    nn.Conv2d(half_channel, half_channel, kernel_size = 5, stride = 5, padding= 0),
                                    ).to('cuda:0')


        self.hyperDecoder = nn.Sequential(
                                    nn.ConvTranspose2d(half_channel, half_channel, kernel_size = 5, stride = 5, output_padding=0, padding = 0),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(half_channel, half_channel, kernel_size = (3,4), stride = (3,4), output_padding=0, padding = 0),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(half_channel, half_channel, kernel_size = (3,4), stride = (3,4), output_padding=0, padding = 0),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(half_channel, half_channel, kernel_size = 5, stride = 2, output_padding=1, padding = 2),
                                    nn.ReLU(),
                                    #nn.ConvTranspose2d(int(self.args.compressed_channel/2), 2 * self.args.compressed_channel, kernel_size = 5, stride = 2, output_padding=1, padding = 2)
                                    nn.Conv2d(half_channel, int(channel * 2), kernel_size = 3, stride = 1, padding= 1)
                                    ).to('cuda:0')

        self.entropy_bottleneck = EntropyBottleneck(half_channel).to('cuda:0')
        self.gaussian_conditional = GaussianConditional(None).to('cuda:0')

        self.map_classifier_list = nn.ModuleList([self._generate_map_classifier(channel) for i in range(self.args.tau + 1)]).to('cuda:0')

    def _generate_map_classifier(self, channel):
        map_classifier = nn.Sequential(nn.Conv2d(channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False))
        return map_classifier


    def forward(self, imgs, visualize=False):

        imgs = imgs[:,0]

        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        entropy_coding_cost = 0
        z_bits_loss = 0
        feature_bits_loss = 0
        bits_loss = 0

        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:1'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = self.feature_extraction(img_feature)

            
            z = self.hyperEncoder(img_feature)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.hyperDecoder(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)

            # quantization
            if self.training:
                img_feature_hat = img_feature + torch.rand_like(img_feature) - 0.5
            else:
                img_feature_hat = torch.round(img_feature)

            img_feature_likelihoods = GaussianLikelihoodEstimation(img_feature_hat, scales_hat, means=means_hat)

            bits_loss_cam = (torch.log(img_feature_likelihoods).sum() + torch.log(z_likelihoods).sum()) / (-math.log(2))
            bits_loss += bits_loss_cam
            z_bits_loss_cam = (torch.log(z_likelihoods).sum()) / (-math.log(2))
            feature_bits_loss_cam = (torch.log(img_feature_likelihoods).sum()) / (-math.log(2))
            z_bits_loss += z_bits_loss_cam
            feature_bits_loss += feature_bits_loss_cam

            img_feature = F.interpolate(img_feature_hat, self.upsample_shape, mode='bilinear')

            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            world_feature = kornia.geometry.transform.warp_perspective(img_feature.to('cuda:0'), proj_mat, self.reducedgrid_shape)
            world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)

        world_features = world_features.to('cuda:0')


        map_result_list = []

        for i in range(self.args.tau + 1):
            map_result = self.map_classifier_list[i](world_features)
            map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
            map_result_list.append(map_result.unsqueeze(1))

        map_result_list = torch.cat(map_result_list, dim = 1)

        bits_loss = bits_loss / 8 / 1024 # KB

        return map_result_list, bits_loss

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret

