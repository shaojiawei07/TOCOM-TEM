import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, beta=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.beta = beta

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        gt_losses = 0 
        bits_losses = 0

        taret_bit = self.model.args.target_bit

        target_bits_loss = torch.FloatTensor([taret_bit]).to('cuda:0')

        precision_s, recall_s = AverageMeter(), AverageMeter()
        for batch_idx, (data, map_gt, _) in enumerate(data_loader):

            optimizer.zero_grad()
            map_res_list, bits_loss = self.model(data)
            map_res = map_res_list[:,0]
            loss = 0

            gt_loss_first = self.criterion(map_res_list[:,0], map_gt[:,0].to(map_res_list[:,0].device), data_loader.dataset.map_kernel)
            gt_weighted_loss = gt_loss_first

            for i in range(1,self.model.args.tau + 1):

                gt_loss = self.criterion(map_res_list[:,i], map_gt[:,i].to(map_res_list[:,i].device), data_loader.dataset.map_kernel)
                gt_weighted_loss = gt_weighted_loss + gt_loss * (0.5 ** i)

            loss = gt_weighted_loss + torch.max(bits_loss, target_bits_loss) * self.beta


            loss.backward()
            optimizer.step()

            losses += loss.item()
            gt_losses += gt_loss_first.item()
            bits_losses += bits_loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:

                print('Epoch: {}, batch: {}, loss: {:.6f}, gt_losses: {:.6f}, communication cost: {:.2f} KB'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), gt_losses / (batch_idx + 1), bits_losses/(batch_idx + 1)))


        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        print("res_fpath",res_fpath)
        print("gt_fpath",gt_fpath)

        self.model.eval()
        losses = 0
        bits_losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        #t0 = time.time()
        output_map_res_statistic = 0
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, frame) in enumerate(data_loader):
            with torch.no_grad():
                map_res_list, bits_loss = self.model(data)
                map_res = map_res_list[:,0]
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)

                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = self.criterion(map_res_list[:,0], map_gt[:,0].to(map_res_list[:,0].device), data_loader.dataset.map_kernel)

            losses += loss.item()
            bits_losses += bits_loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        moda = 0
        
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     data_loader.dataset.base.__name__)

            print('moda: {:.2f}%, modp: {:.2f}%, precision: {:.2f}%, recall: {:.2f}%'.
                  format(moda, modp, precision, recall))

        print('Communication cost: {:.2f} KB'.format(bits_losses / (len(data_loader))))

        return losses / len(data_loader), precision_s.avg * 100, moda


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
