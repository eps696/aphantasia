# coding: UTF-8
import os
import math
import time
from imageio import imread, imsave
import cv2
import numpy as np
import collections
import scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline as CubSpline
import matplotlib.pyplot as plt
from kornia.filters.sobel import spatial_gradient

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

def old_torch():
    ver = [int(i) for i in torch.__version__.split('.')[:2]]
    return True if (ver[0] < 2 and ver[1] < 8) else False

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
    
def cvshow(img):
    img = np.array(img)
    if img.shape[0] > 720 or img.shape[1] > 1280:
        x_ = 1280 / img.shape[1]
        y_ = 720  / img.shape[0]
        psize = tuple([int(s * min(x_, y_)) for s in img.shape[:2][::-1]])
        img = cv2.resize(img, psize)
    cv2.imshow('t', img[:,:,::-1])
    cv2.waitKey(1)

def checkout(img, fname=None, verbose=False):
    img = np.transpose(np.array(img)[:,:,:], (1,2,0))
    if verbose is True:
        cvshow(img)
    if fname is not None:
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        imsave(fname, img)

def save_cfg(args, dir='./', file='config.txt'):
    if dir != '':
        os.makedirs(dir, exist_ok=True)
    try: args = vars(args)
    except: pass
    if file is None:
        print_dict(args)
    else:
        with open(os.path.join(dir, file), 'w') as cfg_file:
            print_dict(args, cfg_file)

def print_dict(dict, file=None, path="", indent=''):
    for k in sorted(dict.keys()):
        if isinstance(dict[k], collections.abc.Mapping):
            if file is None:
                print(indent + str(k))
            else:
                file.write(indent + str(k) + ' \n')
            path = k if path=="" else path + "->" + k
            print_dict(dict[k], file, path, indent + '   ')
        else:
            if file is None:
                print('%s%s: %s' % (indent, str(k), str(dict[k])))
            else:
                file.write('%s%s: %s \n' % (indent, str(k), str(dict[k])))

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

def smoothstep(x, NN=1, xmin=0., xmax=1.):
    N = math.ceil(NN)
    x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    result = 0
    for n in range(0, N+1):
         result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n
    result *= x**(N+1)
    if NN != N: result = (x + result) / 2
    return result

def slerp(z1, z2, num_steps=None, x=None, smooth=0.5):
    z1_norm = z1.norm()
    z2_norm = z2.norm()
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []
    if num_steps is not None:
        xs = [step / (num_steps - 1) for step in range(num_steps)]
    else:
        xs = [x]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interplain = z1 + (z2 - z1) * x
        interp = z1 + (z2_normal - z1) * x
        interp_norm = interp.norm()
        if interp_norm != 0:
            interpol_normal = interplain * (z1_norm / interp_norm)
        vectors.append(interpol_normal)
    return torch.cat(vectors)

def slice_imgs(imgs, count, size=224, transform=None, align='uniform', macro=0.):
    def map(x, a, b):
        return x * (b-a) + a

    rnd_size = torch.rand(count)
    if align == 'central': # normal around center
        rnd_offx = torch.clip(torch.randn(count) * 0.2 + 0.5, 0., 1.)
        rnd_offy = torch.clip(torch.randn(count) * 0.2 + 0.5, 0., 1.)
    else: # uniform
        rnd_offx = torch.rand(count)
        rnd_offy = torch.rand(count)
    
    sz = [img.shape[2:] for img in imgs]
    sz_max = [torch.min(torch.tensor(s)) for s in sz]
    if 'over' in align: # expand frame to sample outside 
        if align == 'overmax':
            sz = [[2*s[0], 2*s[1]] for s in list(sz)]
        else:
            sz = [[int(1.5*s[0]), int(1.5*s[1])] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], sz[i], type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        sz_max_i = sz_max[i]
        for c in range(count):
            sz_min_i = 0.9*sz_max[i] if torch.rand(1) < macro else size
            csize = map(rnd_size[c], sz_min_i, sz_max_i).int()
            offsetx = map(rnd_offx[c], 0, sz[i][1] - csize).int()
            offsety = map(rnd_offy[c], 0, sz[i][0] - csize).int()
            cut = img[:, :, offsety:offsety + csize, offsetx:offsetx + csize]
            cut = F.interpolate(cut, (size,size), mode='bicubic', align_corners=True) # bilinear
            if transform is not None: 
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced

def derivat(img, mode='sobel'):
    if mode == 'scharr': 
        # https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators
        k_scharr = torch.Tensor([[[-0.183,0.,0.183], [-0.634,0.,0.634], [-0.183,0.,0.183]], [[-0.183,-0.634,-0.183], [0.,0.,0.], [0.183,0.634,0.183]]])
        k_scharr = k_scharr.unsqueeze(1).tile((1,3,1,1)).cuda()
        return 0.2 * torch.mean(torch.abs(F.conv2d(img, k_scharr)))
    elif mode == 'sobel':
        # https://kornia.readthedocs.io/en/latest/filters.html#edge-detection
        return torch.mean(torch.abs(spatial_gradient(img)))
    else: # trivial hack
        dx = torch.mean(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))
        dy = torch.mean(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
        return 0.5 * (dx+dy)

def dot_compare(v1, v2, cossim_pow=0):
    dot = (v1 * v2).sum()
    mag = torch.sqrt(torch.sum(v2**2))
    cossim = dot/(1e-6 + mag)
    return dot * cossim ** cossim_pow

def sim_func(v1, v2, type=None):
    if type is not None and 'mix' in type: # mixed
        coss = torch.cosine_similarity(v1, v2, dim=-1).mean()
        v1 = F.normalize(v1, dim=-1)
        v2 = F.normalize(v2, dim=-1)
        spher = torch.abs((v1 - v2).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()
        return coss - 0.25 * spher
    elif type is not None and 'spher' in type: # spherical
        # from https://colab.research.google.com/drive/1ED6_MYVXTApBHzQObUPaaMolgf9hZOOF
        v1 = F.normalize(v1, dim=-1)
        v2 = F.normalize(v2, dim=-1)
        # return 1 - torch.abs((v1 - v2).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()
        return (v1 - v2).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    elif type is not None and 'ang' in type: # angular
        # return 1 - torch.acos(torch.cosine_similarity(v1, v2, dim=-1).mean()) / np.pi
        return 1 - torch.acos(torch.cosine_similarity(v1, v2, dim=-1)).mean() / np.pi
    elif type is not None and 'dot' in type: # dot compare cossim from lucent inversion
        return dot_compare(v1, v2, cossim_pow=1) # decrease pow if nan (black output)
    else:
        return torch.cosine_similarity(v1, v2, dim=-1).mean()

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def get_z(shape, rnd, uniform=False):
    if uniform:
        return rnd.uniform(0., 1., shape)
    else:
        return rnd.randn(*shape) # *x unpacks tuple/list to sequence

def smoothstep(x, NN=1., xmin=0., xmax=1.):
    N = math.ceil(NN)
    x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    result = 0
    for n in range(0, N+1):
         result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n
    result *= x**(N+1)
    if NN != N: result = (x + result) / 2
    return result

def lerp(z1, z2, num_steps, smooth=0.): 
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interpol = z1 + (z2 - z1) * x
        vectors.append(interpol)
    return np.array(vectors)

# interpolate on hypersphere
def slerp_np(z1, z2, num_steps, smooth=0.):
    z1_norm = np.linalg.norm(z1)
    z2_norm = np.linalg.norm(z2)
    z2_normal = z2 * (z1_norm / z2_norm)
    vectors = []
    xs = [step / (num_steps - 1) for step in range(num_steps)]
    if smooth > 0: xs = [smoothstep(x, smooth) for x in xs]
    for x in xs:
        interplain = z1 + (z2 - z1) * x
        interp = z1 + (z2_normal - z1) * x
        interp_norm = np.linalg.norm(interp)
        interpol_normal = interplain * (z1_norm / interp_norm)
        # interpol_normal = interp * (z1_norm / interp_norm)
        vectors.append(interpol_normal)
    return np.array(vectors)

def cublerp(points, steps, fstep, looped=True):
    keys = np.array([i*fstep for i in range(steps)] + [steps*fstep])
    last_pt_num = 0 if looped is True else -1
    points = np.concatenate((points, np.expand_dims(points[last_pt_num], 0)))
    cspline = CubSpline(keys, points)
    return cspline(range(steps*fstep+1))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    
def latent_anima(shape, frames, transit, key_latents=None, smooth=0.5, uniform=False, cubic=False, gauss=False, start_lat=None, seed=None, looped=True, verbose=False):
    if key_latents is None:
        transit = int(max(1, min(frames//2, transit)))
    steps = max(1, math.ceil(frames / transit))
    log = ' timeline: %d steps by %d' % (steps, transit)

    if seed is None:
        seed = np.random.seed(int((time.time()%1) * 9999))
    rnd = np.random.RandomState(seed)
    
    # make key points
    if key_latents is None:
        key_latents = np.array([get_z(shape, rnd, uniform) for i in range(steps)])
    if start_lat is not None:
        key_latents[0] = start_lat

    latents = np.expand_dims(key_latents[0], 0)
    
    # populate lerp between key points
    if transit == 1:
        latents = key_latents
    else:
        if cubic:
            latents = cublerp(key_latents, steps, transit, looped)
            log += ', cubic'
        else:
            for i in range(steps):
                zA = key_latents[i]
                lat_num = (i+1)%steps if looped is True else min(i+1, steps-1)
                zB = key_latents[lat_num]
                if uniform is True:
                    interps_z = lerp(zA, zB, transit, smooth=smooth)
                else:
                    interps_z = slerp_np(zA, zB, transit, smooth=smooth)
                latents = np.concatenate((latents, interps_z))
    latents = np.array(latents)
    
    if gauss:
        lats_post = gaussian_filter(latents, [transit, 0, 0], mode="wrap")
        lats_post = (lats_post / np.linalg.norm(lats_post, axis=-1, keepdims=True)) * math.sqrt(np.prod(shape))
        log += ', gauss'
        latents = lats_post
        
    if verbose: print(log)
    if latents.shape[0] > frames: # extra frame
        latents = latents[1:]
    return latents
    
