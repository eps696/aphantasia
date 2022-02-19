### edited from the code by https://twitter.com/deKxi

import os
import sys
import cv2
from imageio import imsave
import numpy as np
import PIL

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from kornia.enhance import equalize

from aphantasia.utils import triangle_blur
from .adabins import model_io
from .adabins.models import UnetAdaptiveBins

class InferenceHelper:
    def __init__(self, model_path='models/AdaBins_nyu.pt', device='cuda:0', multirun=False):
        self.device = device
        self.min_depth = 1e-3
        self.max_depth = 10 # *50 ?
        self.saving_factor = 1000  # used to save in 16 bit
        self.multirun = multirun
        model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
        model, _, _ = model_io.load_checkpoint(model_path, model)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict(self, image):
        _, pred = self.model(image)
        pred = torch.clip(pred, self.min_depth, self.max_depth)
        if self.multirun is True:
            # Take average of original and mirrors
            pred_hw = self.model(torch.flip(image, [-2]))[-1] # Flip vertically
            pred_hw = torch.clip(torch.flip(pred_hw, [-2]), self.min_depth, self.max_depth)
            pred_lr = self.model(torch.flip(image, [-1]))[-1] # Flip horizontally
            pred_lr = torch.clip(torch.flip(pred_lr, [-1]), self.min_depth, self.max_depth)
            pred = (pred + pred_hw + pred_lr) / 3.
        final = F.interpolate(pred, image.shape[-2:], mode='bicubic', align_corners=True)
        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[torch.isinf(final)] = self.max_depth
        final[torch.isnan(final)] = self.min_depth
        return final

def save_img(img, fname=None):
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1,2,0))  
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    if fname is not None:
        imsave(fname, np.array(img))

def init_adabins(size, model_path='models/AdaBins_nyu.pt', mask_path='lib/adabins/mask.jpg', mask_blur=33, tridepth=False):
    infer_helper = InferenceHelper(model_path, multirun=tridepth)
    # mask for blending multi-crop depth 
    masksize = (830, 500) # it doesn't have to be this exact number, this is just the max for what works at 16:9 for each crop
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, masksize)
    mask = cv2.GaussianBlur(mask, (mask_blur,mask_blur),0)
    mask = cv2.resize(mask, (size[1]//2, size[0]//2)) / 255.
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
    return infer_helper, mask

def resize(img, size):
    return F.interpolate(img, size, mode='bicubic', align_corners=True).float().cuda()
    
def add_equal(img, x):
    img_norm = img / torch.max(img)
    img_eq = equalize(img_norm)
    return torch.lerp(img_norm, img_eq, x)

def denorm(x):
    std_ = torch.as_tensor((0.229, 0.224, 0.225)).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()
    mean_ = torch.as_tensor((0.485, 0.456, 0.406)).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()
    return torch.clip(x * std_ + mean_, 0., 1.)

def depthwarp(img, infer_helper, mask, size, strength, centre=[0,0], midpoint=0.5, equalhist=0.5, save_path=None, save_num=0, multicrop=True):
    ch, cw = size
    _, _, H, W = img.shape
    centre = torch.as_tensor(centre).cuda() # centre/origin point for depth extrusion

    # Resize down for inference
    if H < W: # 500p on either dimension was the limit I found for AdaBins
        r = 500 / float(H)
        dim = (500, int(W*r))
    else:
        r = 500 / float(W)
        dim = (int(H*r), 500)
    # image = denorm(resize(triangle_blur(img, 5, 2), dim)) # blur 
    image = denorm(resize(torch.lerp(img, triangle_blur(img, 5, 2), 0.3), dim)) # blur lerp
    
    pred_depth = infer_helper.predict(image)
    pred_depth = resize(pred_depth, (H,W)) # Resize back to original before optional crops

    if multicrop: 
        # Splitting the image into separate crops, probably inefficiently
        TL = resize(TF.crop(resize(image, (H,W)), top=0,  left=0,  height=ch, width=cw), dim)
        TR = resize(TF.crop(resize(image, (H,W)), top=0,  left=cw, height=ch, width=cw), dim)
        BL = resize(TF.crop(resize(image, (H,W)), top=ch, left=0,  height=ch, width=cw), dim)
        BR = resize(TF.crop(resize(image, (H,W)), top=ch, left=cw, height=ch, width=cw), dim)
        # Inference on crops
        pred_TL = infer_helper.predict(TL)
        pred_TR = infer_helper.predict(TR)
        pred_BL = infer_helper.predict(BL)
        pred_BR = infer_helper.predict(BR)

        # Combining / blending the crops with the original [so so solution]
        clone = pred_depth.clone().detach()
        TL = clone[:, :, 0:ch, 0:cw]         * (1-mask) + resize(pred_TL,(ch,cw)) * mask
        TR = clone[:, :, 0:ch, cw:cw+cw]     * (1-mask) + resize(pred_TR,(ch,cw)) * mask
        BL = clone[:, :, ch:ch+ch, 0:cw]     * (1-mask) + resize(pred_BL,(ch,cw)) * mask
        BR = clone[:, :, ch:ch+ch, cw:cw+cw] * (1-mask) + resize(pred_BR,(ch,cw)) * mask
        clone[:, :, 0:ch, 0:cw] = TL
        clone[:, :, 0:ch, cw:cw+cw] = TR
        clone[:, :, ch:ch+ch, 0:cw] = BL
        clone[:, :, ch:ch+ch, cw:cw+cw] = BR
        
        pred_depth = (pred_depth + clone) / 2.

    if equalhist != 0:
        # Clipping the very end values that often throw the histogram equalize balance off which can mitigate rescales negative effects
        # gmin = torch.quantile(pred_depth, 0.05)
        # gmax = torch.quantile(pred_depth, 0.92)
        # pred_depth = torch.clip(pred_depth, gmin, gmax)
        pred_depth = add_equal(pred_depth, equalhist)

    dtensor = 1. - (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) # Depth is reversed, hence 1-x
    # del image, pred_depth

    if save_path is not None: # Save depth map out, currently its as its own image but it could just be added as an alpha channel to main image
        out_depth = dtensor.detach().clone().cpu().squeeze(0)
        save_img(out_depth, os.path.join(save_path, '%05d.jpg' % save_num))

    dtensor = dtensor.squeeze(0)

    # Building the coordinates
    xx = torch.linspace(-1, 1, W)
    yy = torch.linspace(-1, 1, H)
    gy, gx = torch.meshgrid(yy, xx)
    
    # Apply depth warp
    grid = torch.stack([gx, gy], dim=-1).cuda()
    d = centre - grid
    d_sum = dtensor[0]
    # Adjust midpoint / move direction
    d_sum = d_sum - torch.max(d_sum) * midpoint
    grid += d * d_sum.unsqueeze(-1) * strength
    img = F.grid_sample(img, grid.unsqueeze(0), align_corners=True, padding_mode='reflection')

    # Apply simple lens distortion to mitigate the "stretching" that appears in the periphery
    grid = torch.stack([gx, gy], dim=-1).cuda()
    lens_distortion = torch.sqrt((d**2).sum(axis=-1)).cuda()
    grid += d * lens_distortion.unsqueeze(-1) * strength * 0.31
    img = F.grid_sample(img, grid.unsqueeze(0), align_corners=True, padding_mode='reflection')

    return img
