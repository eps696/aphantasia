import os
# import warnings
# warnings.filterwarnings("ignore")
import argparse
import numpy as np
import cv2
import shutil
from googletrans import Translator, constants

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import ssim

from progress_bar import ProgressBar
from utils import pad_up_to, basename, img_list, img_read

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_img', default=None, help='input image')
parser.add_argument('-t', '--in_txt', default=None, help='input text')
parser.add_argument('-t2', '--in_txt2', default=None, help='input text for small details')
parser.add_argument('-t0', '--in_txt0', default=None, help='input text to subtract')
parser.add_argument('-o', '--out_dir', default='_out')
parser.add_argument('--size', '-s', default='1280-720', help='Output resolution')
parser.add_argument('--resume', '-r', default=None, help='Path to saved params, to resume from')
parser.add_argument('--fstep', default=1, type=int, help='Saving step')
parser.add_argument('--translate', '-tr', action='store_true')
parser.add_argument('--save_pt', action='store_true', help='Save FFT params for further use')
parser.add_argument('--verbose', '-v', action='store_true')
# training
parser.add_argument('--steps', default=500, type=int, help='Total iterations')
parser.add_argument('--samples', default=128, type=int, help='Samples to evaluate')
parser.add_argument('--lrate', default=0.05, type=float, help='Learning rate')
# parser.add_argument('--contrast', default=1., type=float)
parser.add_argument('--uniform', '-u', default=True, help='Extra padding to avoid central localization')
parser.add_argument('--sync', '-c', default=0, type=float, help='Sync output to input image')
parser.add_argument('--invert', '-n', action='store_true', help='Invert criteria')
parser.add_argument('--dual', '-d', action='store_true', help='Use both CLIP models at once')
a = parser.parse_args()

if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
if len(a.size)==1: a.size = a.size * 2

### FFT from Lucent library ###  https://github.com/greentfrapp/lucent

def to_valid_rgb(image_f, decorrelate=True):
    def inner(*args):
        image = image_f(*args)
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)
    return inner
    
def _linear_decorrelate_color(tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_permute = tensor.permute(0,2,3,1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
    tensor = t_permute.permute(0,3,1,2)
    return tensor

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

def pixel_image(shape, sd=2.):
    tensor = (torch.randn(*shape) * sd).cuda().requires_grad_(True)
    return [tensor], lambda: tensor

# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

def fft_image(shape, sd=0.01, decay_power=1.0):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components

    if a.resume is None:
        spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).cuda().requires_grad_(True)
    elif os.path.isfile(a.resume):
        spectrum_real_imag_t = torch.load(a.resume)[0].cuda().requires_grad_(True)
        print(' resuming from:', a.resume)
        print(spectrum_real_imag_t.shape)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].cuda()

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        image = image * 1.33 / image.std()
        # image = image * a.contrast
        return image
    return [spectrum_real_imag_t], inner

# utility functions

def cvshow(img):
    img = np.array(img)
    if img.shape[0] > 720 or img.shape[1] > 1280:
        x_ = 1280 / img.shape[1]
        y_ = 720  / img.shape[0]
        psize = tuple([int(s * min(x_, y_)) for s in img.shape[:2][::-1]])
        img = cv2.resize(img, psize)
    cv2.imshow('t', img[:,:,::-1])
    cv2.waitKey(100)

def checkout(img, fname=None):
    img = np.transpose(np.array(img)[:,:,:], (1,2,0))
    if a.verbose is True:
        cvshow(img)
    if fname is not None:
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        cv2.imwrite(fname, img[:,:,::-1])

def slice_imgs(imgs, count, transform=None, uniform=False, micro=None):
    def map(x, a, b):
        return x * (b-a) + a

    rnd_size = torch.rand(count)
    if uniform is True:
        rnd_offx = torch.rand(count)
        rnd_offy = torch.rand(count)
    else: # normal around center
        rnd_offx = torch.clip(torch.randn(count) * 0.23 + 0.5, 0, 1) 
        rnd_offy = torch.clip(torch.randn(count) * 0.23 + 0.5, 0, 1)
    
    sz = [img.shape[2:] for img in imgs]
    sz_min = [torch.min(torch.tensor(s)) for s in sz]
    if uniform is True:
        sz = [[2*s[0], 2*s[1]] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], sz[i], type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        for c in range(count):
            if micro is True: # both scales, micro mode
                csize = map(rnd_size[c], 64, max(224, 0.25*sz_min[i])).int()
            elif micro is False: # both scales, macro mode
                csize = map(rnd_size[c], 0.5*sz_min[i], 0.98*sz_min[i]).int()
            else: # single scale
                csize = map(rnd_size[c], 112, 0.98*sz_min[i]).int()
            offsetx = map(rnd_offx[c], 0, sz[i][1] - csize).int()
            offsety = map(rnd_offy[c], 0, sz[i][0] - csize).int()
            cut = img[:, :, offsety:offsety + csize, offsetx:offsetx + csize]
            cut = torch.nn.functional.interpolate(cut, (224,224), mode='bicubic', align_corners=False) # bilinear
            if transform is not None: 
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced


def main():

    def train(i):
        loss = 0
        
        img_out = image_f()

        micro = None if a.in_txt2 is None else False
        imgs_sliced = slice_imgs([img_out], a.samples, norm_in, a.uniform, micro=micro)
        out_enc = model_vit.encode_image(imgs_sliced[-1])
        if a.dual is True: # use both clip models
            out_enc = torch.cat((out_enc, model_rn.encode_image(imgs_sliced[-1])), 1)
        if a.in_img is not None and os.path.isfile(a.in_img): # input image
            loss += sign * 100*torch.cosine_similarity(img_enc, out_enc, dim=-1).mean()
        if a.in_txt is not None: # input text
            loss += sign * 100*torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()
        if a.in_txt0 is not None: # subtract text
            loss += -sign * 100*torch.cosine_similarity(txt_enc0, out_enc, dim=-1).mean()
        if a.sync > 0 and a.in_img is not None and os.path.isfile(a.in_img): # image composition
            loss *= 1. + a.sync * (a.steps/(i+1) * ssim_loss(img_out, img_in) - 1)
        if a.in_txt2 is not None: # input text for micro details
            imgs_sliced = slice_imgs([img_out], a.samples, norm_in, a.uniform, micro=True)
            out_enc2 = model_vit.encode_image(imgs_sliced[-1])
            if a.dual is True:
                out_enc2 = torch.cat((out_enc2, model_rn.encode_image(imgs_sliced[-1])), 1)
            loss += sign * 100*torch.cosine_similarity(txt_enc2, out_enc2, dim=-1).mean()
            del out_enc2; torch.cuda.empty_cache()

        del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()
        assert not isinstance(loss, int), ' Loss not defined, check the inputs'
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % a.fstep == 0:
            with torch.no_grad():
                img = image_f().cpu().numpy()[0]
            checkout(img, os.path.join(tempdir, '%03d.jpg' % (i // a.fstep)))
            pbar.upd()

    # Load CLIP models
    model_vit, _ = clip.load('ViT-B/32')
    if a.dual is True:
        print(' using dual-model optimization')
        model_rn, _ = clip.load('RN50', path='models/RN50.pt ')
        a.samples = a.samples // 2
            
    norm_in = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    out_name = []
    if a.in_img is not None and os.path.isfile(a.in_img):
        print(' ref image:', basename(a.in_img))
        img_in = torch.from_numpy(img_read(a.in_img)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
        img_in = img_in[:,:3,:,:] # fix rgb channels
        in_sliced = slice_imgs([img_in], a.samples, transform=norm_in, uniform=a.uniform)[0]
        img_enc = model_vit.encode_image(in_sliced).detach().clone()
        if a.dual is True:
            img_enc = torch.cat((img_enc, model_rn.encode_image(in_sliced).detach().clone()), 1)
        if a.sync > 0:
            ssim_loss = ssim.SSIM(window_size = 11)
            img_in = F.interpolate(img_in, a.size).float()
        else:
            del img_in
        del in_sliced; torch.cuda.empty_cache()
        out_name.append(basename(a.in_img).replace(' ', '_'))

    if a.in_txt is not None:
        print(' ref text: ', basename(a.in_txt))
        if a.translate:
            translator = Translator()
            a.in_txt = translator.translate(a.in_txt, dest='en').text
            print(' translated to:', a.in_txt) 
        tx = clip.tokenize(a.in_txt).cuda()
        txt_enc = model_vit.encode_text(tx).detach().clone()
        if a.dual is True:
            txt_enc = torch.cat((txt_enc, model_rn.encode_text(tx).detach().clone()), 1)
        out_name.append(basename(a.in_txt).replace(' ', '_').replace('\\', '_').replace('/', '_'))

    if a.in_txt2 is not None:
        print(' micro text:', basename(a.in_txt2))
        # a.samples = int(a.samples * 0.9)
        if a.translate:
            translator = Translator()
            a.in_txt2 = translator.translate(a.in_txt2, dest='en').text
            print(' translated to:', a.in_txt2) 
        tx2 = clip.tokenize(a.in_txt2).cuda()
        txt_enc2 = model_vit.encode_text(tx2).detach().clone()
        if a.dual is True:
            txt_enc2 = torch.cat((txt_enc2, model_rn.encode_text(tx2).detach().clone()), 1)
        out_name.append(basename(a.in_txt2).replace(' ', '_').replace('\\', '_').replace('/', '_'))

    if a.in_txt0 is not None:
        print(' subtract text:', basename(a.in_txt0))
        # a.samples = int(a.samples * 0.9)
        if a.translate:
            translator = Translator()
            a.in_txt0 = translator.translate(a.in_txt0, dest='en').text
            print(' translated to:', a.in_txt0) 
        tx0 = clip.tokenize(a.in_txt0).cuda()
        txt_enc0 = model_vit.encode_text(tx0).detach().clone()
        if a.dual is True:
            txt_enc0 = torch.cat((txt_enc0, model_rn.encode_text(tx0).detach().clone()), 1)
        out_name.append('off-' + basename(a.in_txt0).replace(' ', '_').replace('\\', '_').replace('/', '_'))

    params, image_f = fft_image([1, 3, *a.size])
    image_f = to_valid_rgb(image_f)

    optimizer = torch.optim.Adam(params, a.lrate) # pixel 1, fft 0.05
    sign = 1. if a.invert is True else -1.

    print(' samples:', a.samples)
    sfx = ''
    sfx += '-d' if a.dual    is True else ''
    sfx += '-c%.2g'%a.sync if a.sync > 0 else ''
    out_name = '-'.join(out_name) + sfx
    tempdir = os.path.join(a.out_dir, out_name)
    os.makedirs(tempdir, exist_ok=True)

    pbar = ProgressBar(a.steps // a.fstep)
    for i in range(a.steps):
        train(i)

    os.system('ffmpeg -v warning -y -i %s\%%03d.jpg "%s.mp4"' % (tempdir, os.path.join(a.out_dir, out_name)))
    shutil.copy(img_list(tempdir)[-1], os.path.join(a.out_dir, '%s-%d.jpg' % (out_name, a.steps)))
    if a.save_pt is True:
        torch.save(params, '%s.pt' % os.path.join(a.out_dir, out_name))

if __name__ == '__main__':
    main()
