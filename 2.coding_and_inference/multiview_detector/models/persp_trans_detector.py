import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from multiview_detector.models.resnet import resnet18
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
        self.tau_2 = args.tau_2
        self.tau_1 = args.tau_1
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

        self.channel = 8 #self.args.compressed_channel
        self.channel_factor = 4

        self.feature_extraction = nn.Conv2d(512, self.channel, 1).to("cuda:0")

        self.temporal_entropy_model = nn.Sequential(
                                    nn.Conv2d(self.tau_2 * self.channel, 64, kernel_size = 3, stride = 1, padding= 1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding= 1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 2 * self.channel, kernel_size = 3, stride = 1, padding= 1),
                                    ).to('cuda:0')

        self.temporal_fusion_module = nn.Sequential(nn.Conv2d(self.channel * self.num_cam * (self.tau_1 + 1 )+ 2, 512, 3, padding=1), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')

        pretrained_model_path = self.args.model_path

        model_dict = torch.load(pretrained_model_path)

        base_pt1_dict = {k[9:]:v for k,v in model_dict.items() if k[:8] == "base_pt1" }
        base_pt2_dict = {k[9:]:v for k,v in model_dict.items() if k[:8] == "base_pt2" }
        feature_extraction_dict = {k[19:]:v for k,v in model_dict.items() if k[:18] == "feature_extraction" }

        self.base_pt1.load_state_dict(base_pt1_dict)
        self.base_pt2.load_state_dict(base_pt2_dict)
        self.feature_extraction.load_state_dict(feature_extraction_dict)

        for param in self.base_pt1.parameters():
            param.requires_grad = False
        for param in self.base_pt2.parameters():
            param.requires_grad = False
        for param in self.feature_extraction.parameters():
            param.requires_grad = False


    def feature_extraction_step(self, imgs):

        B, N, C, H, W = imgs.shape

        imgs = torch.reshape(imgs,(B * N, C, H, W))

        img_feature = self.base_pt1(imgs.to('cuda:1'))
        img_feature = self.base_pt2(img_feature.to('cuda:0'))
        img_feature = self.feature_extraction(img_feature)
        img_feature = torch.round(img_feature)


        _, C, H, W = img_feature.shape

        img_feature = torch.reshape(img_feature,(B, N, C, H, W))

        return img_feature # (B,N,channel,90,160)


    def forward(self, imgs_list, visualize=False):

        imgs_list_feature = []
        B, T ,N, C, H, W = imgs_list.shape
        assert N == self.num_cam
        world_features = []
        # feature_bits_loss = 0
        # bits_loss = 0
        tau = max(self.tau_1, self.tau_2)


        for i in range(tau + 1):
            imgs  = imgs_list[:,i]
            imgs_feature = self.feature_extraction_step(imgs) # (B,N,channel,90,160)
            imgs_feature = imgs_feature.unsqueeze(dim=1) # (B, 1, N,channel,90,160)
            imgs_list_feature.append(imgs_feature)
        imgs_list_feature = torch.cat(imgs_list_feature, dim = 1) # (B, T, N,channel,90,160)
        #print("size", imgs_list_feature.size())
        assert T == imgs_list_feature.size()[1]

        to_be_transmitted_feature = imgs_list_feature[:,self.tau_2] # (B, N, channel, 90, 160)

        conditional_features = imgs_list_feature[:,:self.tau_2] # (B, self.tau2, N, channel 90, 160)
        conditional_features = torch.swapaxes(conditional_features, 1, 2) # (B, N, self.tau2, channel 90, 160)
        conditional_features = torch.reshape(conditional_features, (B,N, self.tau_2 * self.channel, 90, 160))
        conditional_features = torch.reshape(conditional_features, (B * N, self.tau_2 * self.channel, 90, 160))

        gaussian_params = self.temporal_entropy_model(conditional_features) # (B * N, 2 * self.channel, 90, 160)
        gaussian_params = torch.reshape(gaussian_params, (B, N, 2 * self.channel, 90, 160))
        scales_hat, means_hat = gaussian_params.chunk(2, dim = 2) # (B, N, self.channel, 90, 160)
        feature_likelihoods = GaussianLikelihoodEstimation(to_be_transmitted_feature, scales_hat, means=means_hat)
        bits_loss = (torch.log(feature_likelihoods).sum() / (-math.log(2)))


        feature4prediction = imgs_list_feature[:,-(self.tau_1+1):] # (B, (self.tau_1 +1), N, channel, 90, 160)
        feature4prediction = torch.swapaxes(feature4prediction, 1, 2) # (B, N, (self.tau_1 +1), channel, 90, 160)
        feature4prediction = torch.reshape(feature4prediction, (B,N, (self.tau_1 +1) * self.channel, 90, 160)) # (B, N, (self.tau_1 +1) * channel, 90, 160)
        feature4prediction = torch.reshape(feature4prediction, (B * N, (self.tau_1 +1) * self.channel, 90, 160)) # (B, N, (self.tau_1 +1) * channel, 90, 160)
        feature4prediction = F.interpolate(feature4prediction, size = self.upsample_shape, mode='bilinear') # (B, N, (self.tau_1 +1) * channel, H, W)
        feature4prediction = torch.reshape(feature4prediction, (B, N, (self.tau_1 +1) * self.channel, 270, 480)) # (B, N, (self.tau_1 +1) * channel, 270, 480)

        for cam in range(self.num_cam):
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            world_feature = kornia.geometry.transform.warp_perspective(feature4prediction[:,cam].to('cuda:0'), proj_mat, self.reducedgrid_shape)
            #print(world_feature.size()) # （B, self.tau_1 * channel, H, W）
            world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        world_features = world_features.to('cuda:0')

        map_result = self.temporal_fusion_module(world_features)

        bits_loss = bits_loss / 8 / 1024


        return map_result, bits_loss #0#, imgs_result

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
