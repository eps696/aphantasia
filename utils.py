# coding: UTF-8
import os
from imageio import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

def plot_text(txt, size=224):
    fig = plt.figure(figsize=(1,1), dpi=size)
    fontsize = size//len(txt) if len(txt) < 15 else 8
    plt.text(0.5, 0.5, txt, fontsize=fontsize, ha='center', va='center', wrap=True)
    plt.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img

def txt_clean(txt):
    return txt.translate(str.maketrans(dict.fromkeys(list("\n',.вЂ”|!?/:;\\"), ""))).replace(' ', '_').replace('"', '')

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(path, ext=None, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        if isinstance(ext, list):
            files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ext]
        elif isinstance(ext, str):
            files = [f for f in files if f.endswith(ext)]
        else:
            print(' Unknown extension/type for file list!')
    return sorted([f for f in files if os.path.isfile(f)])

def img_list(path, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['jpg', 'jpeg', 'png', 'ppm', 'tif']]
    return sorted([f for f in files if os.path.isfile(f)])

def img_read(path):
    img = imread(path)
    # 8bit to 256bit
    if (img.ndim == 2) or (img.shape[2] == 1):
        img = np.dstack((img,img,img))
    # rgba to rgb
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img
    
def img_save(path, img, norm=True):
    if norm == True and not np.issubdtype(img.dtype.kind, np.integer): 
        img = (img*255).astype(np.uint8) 
    imsave(path, img)
    
def minmax(x, torch=True):
    if torch:
        mn = torch.min(x).detach().cpu().numpy()
        mx = torch.max(x).detach().cpu().numpy()
    else:
        mn = np.min(x.detach().cpu().numpy())
        mx = np.max(x.detach().cpu().numpy())
    return (mn, mx)

# Tiles an array around two points, allowing for pad lengths greater than the input length
# NB: if symm=True, every second tile is mirrored = messed up in GAN
# adapted from https://discuss.pytorch.org/t/symmetric-padding/19866/3
def tile_pad(xt, padding, symm=False):
    h, w = xt.shape[-2:]
    left, right, top, bottom = padding
 
    def tile(x, minx, maxx):
        rng = maxx - minx
        if symm is True: # triangular reflection
            double_rng = 2*rng
            mod = np.fmod(x - minx, double_rng)
            normed_mod = np.where(mod < 0, mod+double_rng, mod)
            out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        else: # repeating tiles
            mod = np.remainder(x - minx, rng)
            out = mod + minx
        return np.array(out, dtype=x.dtype)

    x_idx = np.arange(-left, w+right)
    y_idx = np.arange(-top, h+bottom)
    x_pad = tile(x_idx, -0.5, w-0.5)
    y_pad = tile(y_idx, -0.5, h-0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return xt[..., yy, xx]

def pad_up_to(x, size, type='centr'):
    sh = x.shape[2:][::-1]
    if list(x.shape[2:]) == list(size): return x
    padding = []
    for i, s in enumerate(size[::-1]):
        if 'side' in type.lower():
            padding = padding + [0, s-sh[i]]
        else: # centr
            p0 = (s-sh[i]) // 2
            p1 = s-sh[i] - p0
            padding = padding + [p0,p1]
    y = tile_pad(x, padding, symm = ('symm' in type.lower()))
    return y

