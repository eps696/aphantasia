### original method & code was by https://twitter.com/deKxi

import logging
logging.getLogger('xformers').setLevel(logging.ERROR) # shutup triton, before torch!

import os
import sys
import cv2
from imageio import imsave
import numpy as np
import PIL

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from aphantasia.utils import triangle_blur
from .any2.dpt import DepthAnythingV2

class InferDepthAny:
    def __init__(self, modtype='B', device=torch.device('cuda')):
        modtype = 'Large' if modtype[0].lower()=='l' else 'Small' if modtype[0].lower()=='s' else 'Base'
        from transformers import AutoModelForDepthEstimation
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-%s-hf" % modtype)
        self.model = model.cuda().eval()

    @torch.no_grad()
    def __call__(self, image):
        image = T.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        depth = self.model(pixel_values=image).predicted_depth.unsqueeze(0)
        return (depth - depth.min()) / (depth.max() - depth.min())

def save_img(img, fname=None):
    if fname is not None:
        img = np.array(img)[:,:,:]
        img = np.transpose(img, (1,2,0))  
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        if img.shape[-1]==1: img = img[:,:,[0,0,0]]
        imsave(fname, np.array(img))

def resize(img, size):
    return F.interpolate(img, size, mode='bicubic', align_corners=True).float().cuda()
    
def grid_warp(img, dtensor, H, W, strength, centre, midpoint, dlens=0.05):
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
    grid_warped = grid + d * d_sum.unsqueeze(-1) * strength
    img = F.grid_sample(img, grid_warped.unsqueeze(0).float(), mode='bilinear', align_corners=True, padding_mode='reflection')

    # Apply simple lens distortion to stretch periphery (instead of sphere wrap)
    lens_distortion = torch.sqrt((d**2).sum(axis=-1)).cuda()
    grid_warped = grid + d * lens_distortion.unsqueeze(-1) * strength * dlens
    img = F.grid_sample(img, grid_warped.unsqueeze(0).float(), mode='bilinear', align_corners=True, padding_mode='reflection')

    return img

def depthwarp(img_t, img, infer_any, strength=0, centre=[0,0], midpoint=0.5, save_path=None, save_num=0, dlens=0.05):
    _, _, H, W = img.shape # [1,3,720,1280]  [0..1]

    res = 518 # 518 on lower dimension for DepthAny
    dim = [res, int(res*W/H)] if H < W else [int(res*H/W), res]
    dim = [x - x % 14 for x in dim]

    image = resize(torch.lerp(img, triangle_blur(img, 5, 2), 0.5), dim) # [1,3,518,910]  [0..1]
    depth = infer_any(image) # [1,1,h,w] 
    depth = depth * torch.flip(infer_any(torch.flip(image, [-1])), [-1]) # enhance depth with mirrored estimation
    depth = resize(depth, (H,W)) # [1,1,H,W]
    
    if save_path is not None: # Save depth map out, currently its as its own image but it could just be added as an alpha channel to main image
        out_depth = depth.detach().clone().cpu().squeeze(0)
        save_img(out_depth, os.path.join(save_path, '%05d.jpg' % save_num))

    img = grid_warp(img_t, depth.squeeze(0), H, W, strength, torch.as_tensor(centre).cuda(), midpoint, dlens)

    return img

