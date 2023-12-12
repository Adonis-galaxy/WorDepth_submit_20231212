import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def pixel_unshuffle(fm, r):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    b, c, h, w = fm.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
    fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(b, out_channel, out_h, out_w)

    return fm_prime


class VarLoss(nn.Module):
    def __init__(self, depth_channel, feat_channel):
        super(VarLoss, self).__init__()

        self.att = nn.Sequential(
                nn.Conv2d(feat_channel, depth_channel, kernel_size=3, padding=1),
                nn.Sigmoid())

        self.post = nn.Conv2d(depth_channel, 2, kernel_size=3, padding=1)

        self.r = 10  # repeat sample

    def forward(self, x, d, gts):
        loss = 0.0
        for i in range(self.r):
            loss = loss + self.single(x, d, gts)

        return loss / self.r

    def single(self, feat, d, gts):

        ts_shape = d.shape[2:]
        gt = gts.clone()
        #gt = gts.unsqueeze(1)
        #print(gt.shape)
        os_shape = gt.shape[2:]
        n, c, h, w = d.shape

        reshaped_gt, indices = self.random_pooling(gt, ts_shape)

        bias_x = os_shape[1] // ts_shape[1] // 2
        bias_y = os_shape[0] // ts_shape[0] // 2 * os_shape[1]

        indices = indices + bias_x + bias_y
        ind_x = (indices % os_shape[1]).to(d.dtype) / os_shape[1]
        ind_y = (indices // os_shape[1]).to(d.dtype) / os_shape[0]

        ind_x = 2 * (ind_x - 0.5)
        ind_y = 2 * (ind_y - 0.5)
        grid = torch.cat([ind_x, ind_y], 1)
        grid = grid.permute(0, 2, 3, 1)

        feat = F.grid_sample(input=feat, grid=grid, mode='bilinear', align_corners=True)

        att = self.att(feat)

        ds = att * d
        #att = att.permute(0, 2, 3, 1)

        #ds = F.grid_sample(input=d, grid=grid+att, mode='bilinear', align_corners=True)

        ds = self.post(ds)

        loss = self.loss(ds, reshaped_gt)
        return loss
    
    def random_pooling(self, gt_depth, shape):
        #print(gt_depth.shape)
        n, c, h, w = gt_depth.shape
        rand = torch.rand(n, c, h, w, dtype=gt_depth.dtype, device=gt_depth.device)
        mask = gt_depth > 0.1
        rand = rand * mask

        _, indices = F.adaptive_max_pool2d(rand, shape, return_indices=True)

        reshaped_ind = indices.reshape(n, c, -1)
        reshaped_gt = gt_depth.reshape(n, c, h*w)
        reshaped_gt = torch.gather(input=reshaped_gt, dim=-1, index=reshaped_ind)
        reshaped_gt = reshaped_gt.reshape(n, c, indices.shape[2], indices.shape[3])

        reshaped_gt[reshaped_gt < 0.1] = 0
        return reshaped_gt, indices

    def grad(self, image):
        def gradient_y(img):
            gx = torch.log(img[:,:,1:-1,1:-1]+1e-6) - torch.log(img[:,:,2:,1:-1]+1e-6)

            mask = img > 0.1
            mask = torch.logical_and(mask[:,:,1:-1,1:-1], mask[:,:,2:,1:-1])
            return gx, mask

        def gradient_x(img):
            gy = torch.log(img[:,:,1:-1,1:-1]+1e-6) - torch.log(img[:,:,1:-1,2:]+1e-6)

            mask = img > 0.1
            mask = torch.logical_and(mask[:,:,1:-1,1:-1], mask[:,:,1:-1,2:])
            return gy, mask

        image = F.pad(image, (1,1,1,1), 'constant', 0.0)

        image_grad_x, mask_x = gradient_x(image)
        image_grad_y, mask_y = gradient_y(image)

        return image_grad_x, image_grad_y, mask_x, mask_y

    def loss(self, ds, reshaped_gt):

        gx, gy, mx, my = self.grad(reshaped_gt)
        grad_gt = torch.cat([gx, gy], 1)
        grad_mk = torch.cat([mx, my], 1)

        diff = F.smooth_l1_loss(ds, grad_gt, reduce=False, beta=0.01) * grad_mk

        loss_g =  diff.sum() / grad_mk.sum()

        return loss_g

class VD_Loss(nn.Module):
    def __init__(self,  max_depth):
        super(VD_Loss, self).__init__()

        self.max_depth = max_depth
        self.huber_loss=torch.nn.HuberLoss()
        self.w_smooth = 1e-2

    def gradient_yx(self, T):
        '''
        Computes gradients in the y and x directions

        Arg(s):
            T : torch.Tensor[float32]
                N x C x H x W tensor
        Returns:
            torch.Tensor[float32] : gradients in y direction
            torch.Tensor[float32] : gradients in x direction
        '''

        dx = T[:, :, :, :-1] - T[:, :, :, 1:]
        dy = T[:, :, :-1, :] - T[:, :, 1:, :]
        return dy, dx

    def smoothness_loss_func(self, weight, predict, image):
        '''
        Computes the local smoothness loss

        Arg(s):
            weight : torch.Tensor[float32]
                N x 1 x H x W binary mask
            predict : torch.Tensor[float32]
                N x 1 x H x W predictions
            image : torch.Tensor[float32]
                N x 3 x H x W RGB image
        Returns:
            torch.Tensor[float32] : mean local smooth loss
        '''


        predict_dy, predict_dx = self.gradient_yx(predict)
        image_dy, image_dx = self.gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        weight_valid_x=weight[:,:,:,:-1]
        weight_valid_y=weight[:,:,:-1,:]

        smoothness_x = torch.sum(weight_valid_x * weights_x * torch.abs(predict_dx)) / torch.sum(weight_valid_x)
        smoothness_y = torch.sum(weight_valid_y * weights_y * torch.abs(predict_dy)) / torch.sum(weight_valid_y)

        return smoothness_x + smoothness_y

    def forward(self, depth_prediction, reshaped_gt, image):

        loss_depth = torch.abs(depth_prediction-reshaped_gt)


        weight_depth = torch.where(
        (reshaped_gt > 1e-3) * (reshaped_gt < self.max_depth),
        torch.ones_like(loss_depth),
        torch.zeros_like(loss_depth)
        )

        loss_depth = torch.sum(weight_depth * loss_depth) / torch.sum(weight_depth) 

        weight_smooth = 1-weight_depth

        loss_smooth = self.smoothness_loss_func(weight_smooth,depth_prediction,image)

        loss = loss_depth + self.w_smooth * loss_smooth

        return loss
