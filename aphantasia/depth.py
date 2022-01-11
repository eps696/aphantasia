### original code & comments by https://twitter.com/deKxi

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

from .adabins.infer import InferenceHelper

def numpy2tensor(imgArray):
    im = torch.unsqueeze(T.ToTensor()(imgArray), 0)
    return im

def save_img(img, fname=None):
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1,2,0))  
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    if fname is not None:
        imsave(fname, np.array(img))

def init_adabins(size, model_path, mask_path='lib/adabins/mask.jpg', mask_blur=33):
    depth_infer = InferenceHelper(model_path)
    # mask for blending multi-crop depth 
    masksize = (830, 500) # it doesn't have to be this exact number, this is just the max for what works at 16:9 for each crop
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, masksize)
    mask = cv2.GaussianBlur(mask, (mask_blur,mask_blur),0)
    mask = cv2.resize(mask, (size[1]//2, size[0]//2)) / 255.
    return depth_infer, mask

def depthwarp(img, image, infer_helper, mask, size, strength, centre=[0,0], midpoint=0.5, rescale=0, clip_range=0, save_path=None, save_num=0, multicrop=True):
    ch, cw = size
    _, _, H, W = img.shape
    # centre/origin point for the depth extrusion
    centre = torch.as_tensor(centre).cpu()

    # Resize down for inference
    if H < W: # 500p on either dimension was the limit I found for AdaBins
        r = 500 / float(H)
        dim = (int(W * r), 500)
    else:
        r = 500 / float(W)
        dim = (500, int(H * r))
    image = image.resize(dim,3)

    bin_centres, predicted_depth = infer_helper.predict_pil(image)     
    
    # Resize back to original before (optionally) adding the cropped versions
    predicted_depth = cv2.resize(predicted_depth[0][0],(W,H))

    if multicrop: 
        # This code is very jank as I threw it together as a quick proof-of-concept, and it miraculously worked 
        # There's very likely to be some improvements that can be made

        clone = predicted_depth.copy()
        # Splitting the image into separate crops, probably inefficiently
        TL = T.functional.crop(image.resize((H,W),3), top=0, left=0, height=cw, width=ch).resize(dim,3)
        TR = T.functional.crop(image.resize((H,W),3), top=0, left=ch, height=cw, width=ch).resize(dim,3)
        BL = T.functional.crop(image.resize((H,W),3), top=cw, left=0, height=cw, width=ch).resize(dim,3)
        BR = T.functional.crop(image.resize((H,W),3), top=cw, left=ch, height=cw, width=ch).resize(dim,3)

        # Inference on crops
        _, predicted_TL = infer_helper.predict_pil(TL)
        _, predicted_TR = infer_helper.predict_pil(TR)
        _, predicted_BL = infer_helper.predict_pil(BL)
        _, predicted_BR = infer_helper.predict_pil(BR)

        # Rescale will increase per object depth difference, but may cause more depth fluctuations if set too high
        # This likely results in the depth map being less "accurate" to any real world units.. not that it was particularly in the first place lol
        if rescale != 0:
            # Histogram equalize requires a range of 0-255, but I'm recombining later in 0-1 hence this mess
            TL = cv2.addWeighted(cv2.equalizeHist(predicted_TL.astype(np.uint8) * 255) / 255., 1-rescale, predicted_TL.astype(np.uint8),rescale,0)
            TR = cv2.addWeighted(cv2.equalizeHist(predicted_TR.astype(np.uint8) * 255) / 255., 1-rescale, predicted_TR.astype(np.uint8),rescale,0)
            BL = cv2.addWeighted(cv2.equalizeHist(predicted_BL.astype(np.uint8) * 255) / 255., 1-rescale, predicted_BL.astype(np.uint8),rescale,0)
            BR = cv2.addWeighted(cv2.equalizeHist(predicted_BR.astype(np.uint8) * 255) / 255., 1-rescale, predicted_BR.astype(np.uint8),rescale,0)
        # Combining / blending the crops with the original [so so solution]
        TL = clone[0: ch, 0: cw] * (1 - mask) + cv2.resize(predicted_TL[0][0],(cw,ch)) * mask
        TR = clone[0: ch, cw: cw+cw] * (1 - mask) + cv2.resize(predicted_TR[0][0],(cw,ch)) * mask
        BL = clone[ch: ch+ch, 0: cw] * (1 - mask) + cv2.resize(predicted_BL[0][0],(cw,ch)) * mask
        BR = clone[ch: ch+ch, cw: cw+cw] * (1 - mask) + cv2.resize(predicted_BR[0][0],(cw,ch)) * mask

        clone[0: ch, 0: cw] = TL
        clone[0: ch, cw: cw+cw] = TR
        clone[ch: ch+ch, 0: cw] = BL
        clone[ch: ch+ch, cw: cw+cw] = BR
        
        # I'm just multiplying the depths currently, but performing a pixel average is possibly a better idea
        predicted_depth = predicted_depth * clone
        predicted_depth /= np.max(predicted_depth) # Renormalize so we don't blow the image out of range

    #### Generating a new depth map on each frame can sometimes cause temporal depth fluctuations and "popping". 
    #### This part is just some of my experiments trying to mitigate that
    # Dividing by average
    #ave = np.mean(predicted_depth)
    #predicted_depth = np.true_divide(predicted_depth, ave)
    # Clipping the very end values that often throw the histogram equalize balance off which can mitigate rescales negative effects
    gmin = np.percentile(predicted_depth, 0 + clip_range) # 5
    gmax = np.percentile(predicted_depth, 100 - clip_range) # 8
    clipped = np.clip(predicted_depth, gmin, gmax)

    # Depth is reversed, hence the "1 - x"
    predicted_depth = (1 - ((clipped - gmin) / (gmax - gmin))) * 255

    # Rescaling helps emphasise the depth difference but is less "accurate". The amount gets mixed in via lerp
    if rescale != 0:
        rescaled = numpy2tensor(cv2.equalizeHist(predicted_depth.astype(np.uint8)))
        rescaled = T.Resize((H,W))(rescaled.cuda())
    
    # Renormalizing again before converting back to tensor
    predicted_depth = predicted_depth.astype(np.uint8) / np.max(predicted_depth.astype(np.uint8))
    dtensor = numpy2tensor(PIL.Image.fromarray(predicted_depth)).cuda()
    #dtensor = T.Resize((H,W))(dtensor.cuda())

    if rescale != 0: # Mixin amount for rescale, from 0-1
        dtensor = torch.lerp(dtensor, rescaled, rescale)

    if save_path is not None: # Save depth map out, currently its as its own image but it could just be added as an alpha channel to main image
        out_depth = dtensor.detach().clone().cpu().squeeze(0)
        save_img(out_depth, os.path.join(save_path, '%05d.jpg' % save_num))

    dtensor = dtensor.squeeze(0)

    # Building the coordinates, most of this is on CPU since it uses numpy
    xx = torch.linspace(-1, 1, W)
    yy = torch.linspace(-1, 1, H)
    gy, gx = torch.meshgrid(yy, xx)
    grid = torch.stack([gx, gy], dim=-1).cpu()
    d = (centre-grid).cpu()
    # Simple lens distortion that can help mitigate the "stretching" that appears in the periphery
    lens_distortion = torch.sqrt((d**2).sum(axis=-1)).cpu()
    #grid2 = torch.stack([gx, gy], dim=-1)
    d_sum = dtensor[0]

    # Adjust midpoint / move direction
    d_sum = (d_sum - (torch.max(d_sum) * midpoint)).cpu()
    
    # Apply the depth map (and lens distortion) to the grid coordinates
    grid += d * d_sum.unsqueeze(-1) * strength
    del image, bin_centres, predicted_depth

    # Perform the depth warp
    img = torch.nn.functional.grid_sample(img, grid.unsqueeze(0).cuda(), align_corners=True, padding_mode='reflection')
    
    # Reset and perform the lens distortion warp (with reduced strength)
    grid = torch.stack([gx, gy], dim=-1).cpu()
    grid += d * lens_distortion.unsqueeze(-1) * (strength*0.31)
    img = torch.nn.functional.grid_sample(img, grid.unsqueeze(0).cuda(), align_corners=True, padding_mode='reflection')

    return img
