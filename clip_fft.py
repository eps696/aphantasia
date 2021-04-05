import os
# import warnings
# warnings.filterwarnings("ignore")
import argparse
import numpy as np
import cv2
from imageio import imsave
import shutil
from googletrans import Translator, constants

import torch
import torchvision
import torch.nn.functional as F

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sentence_transformers import SentenceTransformer
import pytorch_ssim as ssim

from utils import pad_up_to, basename, img_list, img_read, txt_clean
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

clip_models = ['ViT-B/32', 'RN50', 'RN50x4', 'RN101']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--in_img',  default=None, help='input image')
    parser.add_argument('-t',  '--in_txt',  default=None, help='input text')
    parser.add_argument('-t2', '--in_txt2', default=None, help='input text for small details')
    parser.add_argument('-t0', '--in_txt0', default=None, help='input text to subtract')
    parser.add_argument(       '--out_dir', default='_out')
    parser.add_argument('-s',  '--size',    default='1280-720', help='Output resolution')
    parser.add_argument('-r',  '--resume',  default=None, help='Path to saved FFT snapshots, to resume from')
    parser.add_argument(       '--fstep',   default=1, type=int, help='Saving step')
    parser.add_argument('-tr', '--translate', action='store_true', help='Translate text with Google Translate')
    parser.add_argument('-ml', '--multilang', action='store_true', help='Use SBERT multilanguage model for text')
    parser.add_argument(       '--save_pt', action='store_true', help='Save FFT snapshots for further use')
    parser.add_argument('-v',  '--verbose', default=True, type=bool)
    # training
    parser.add_argument('-m',  '--model',   default='ViT-B/32', choices=clip_models, help='Select CLIP model to use')
    parser.add_argument(       '--steps',   default=200, type=int, help='Total iterations')
    parser.add_argument(       '--samples', default=200, type=int, help='Samples to evaluate')
    parser.add_argument(       '--lrate',   default=0.05, type=float, help='Learning rate')
    parser.add_argument('-p',  '--prog',    action='store_true', help='Enable progressive lrate growth (up to double a.lrate)')
    # tweaks
    parser.add_argument('-o',  '--overscan', action='store_true', help='Extra padding to add seamless tiling')
    parser.add_argument(       '--contrast', default=1., type=float)
    parser.add_argument(       '--colors',  default=1., type=float)
    parser.add_argument('-d',  '--diverse', default=0, type=float, help='Endorse variety (difference between two parallel samples)')
    parser.add_argument('-x',  '--expand',  default=0, type=float, help='Push farther (difference between prev/next samples)')
    parser.add_argument('-n',  '--noise',   default=0, type=float, help='Add noise to suppress accumulation') # < 0.05 ?
    parser.add_argument('-c',  '--sync',    default=0, type=float, help='Sync output to input image')
    parser.add_argument(       '--invert',  action='store_true', help='Invert criteria')
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    if a.in_img is not None and a.sync > 0: a.overscan = True
    a.modsize = 288 if a.model == 'RN50x4' else 224
    if a.multilang is True: a.model = 'ViT-B/32' # sbert model is trained with ViT
    return a

### FFT from Lucent library ###  https://github.com/greentfrapp/lucent

def to_valid_rgb(image_f, colors=1., decorrelate=True):
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype("float32")
    color_correlation_svd_sqrt /= np.asarray([colors, 1., 1.]) # saturate, empirical
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

    def _linear_decorrelate_color(tensor):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        t_permute = tensor.permute(0,2,3,1)
        t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
        tensor = t_permute.permute(0,3,1,2)
        return tensor

    def inner(*args, **kwargs):
        image = image_f(*args, **kwargs)
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)
    return inner
    
def pixel_image(shape, sd=2.):
    tensor = (torch.randn(*shape) * sd).cuda().requires_grad_(True)
    return [tensor], lambda: tensor

# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional frequency and later cut off 1 pixel
    w2 = (w+1)//2 if w%2 == 1 else w//2+1
    fx = np.fft.fftfreq(w)[:w2]
    return np.sqrt(fx * fx + fy * fy)

def fft_image(shape, sd=0.01, decay_power=1.0, resume=None):
    b, ch, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (b, ch) + freqs.shape + (2,) # 2 for imaginary and real components

    if resume is None:
        spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).cuda().requires_grad_(True)
    elif isinstance(resume, str) and os.path.isfile(resume):
        saved = torch.load(resume)
        if isinstance(saved, list): saved = saved[0]
        spectrum_real_imag_t = (saved * sd).cuda().requires_grad_(True)
        # print(' resuming from:', resume, spectrum_real_imag_t.shape)
    else:
        if isinstance(resume, list): resume = resume[0]
        spectrum_real_imag_t = (resume * sd).cuda().requires_grad_(True)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w*h)
    scale = torch.tensor(scale).float()[None, None, ..., None].cuda()

    def inner(shift=None, contrast=1.):
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if shift is not None:
            scaled_spectrum_t += scale * shift
        image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:b, :ch, :h, :w]
        image = image * contrast / image.std() # keep contrast, empirical
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
    cv2.waitKey(1)

def checkout(img, fname=None, verbose=False):
    img = np.transpose(np.array(img)[:,:,:], (1,2,0))
    if verbose is True:
        cvshow(img)
    if fname is not None:
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        imsave(fname, img)

def slice_imgs(imgs, count, size=224, transform=None, overscan=False, micro=None):
    def map(x, a, b):
        return x * (b-a) + a

    rnd_size = torch.rand(count)
    rnd_offx = torch.rand(count)
    rnd_offy = torch.rand(count)
    
    sz = [img.shape[2:] for img in imgs]
    sz_min = [torch.min(torch.tensor(s)) for s in sz]
    if overscan is True:
        sz = [[2*s[0], 2*s[1]] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], sz[i], type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        for c in range(count):
            if micro is True: # both scales, micro mode
                csize = map(rnd_size[c], 64, max(size, 0.25*sz_min[i])).int()
            elif micro is False: # both scales, macro mode
                csize = map(rnd_size[c], 0.5*sz_min[i], 0.98*sz_min[i]).int()
            else: # single scale
                csize = map(rnd_size[c], 112, 0.98*sz_min[i]).int()
            offsetx = map(rnd_offx[c], 0, sz[i][1] - csize).int()
            offsety = map(rnd_offy[c], 0, sz[i][0] - csize).int()
            cut = img[:, :, offsety:offsety + csize, offsetx:offsetx + csize]
            cut = F.interpolate(cut, (size,size), mode='bicubic', align_corners=False) # bilinear
            if transform is not None: 
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced


def main():
    a = get_args()

    prev_enc = 0
    def train(i):
        loss = 0
        
        noise = a.noise * torch.randn(1, 1, *params[0].shape[2:4], 1).cuda() if a.noise > 0 else None
        img_out = image_f(noise)

        micro = None if a.in_txt2 is None else False
        imgs_sliced = slice_imgs([img_out], a.samples, a.modsize, norm_in, a.overscan, micro=micro)
        out_enc = model_clip.encode_image(imgs_sliced[-1])
        if a.diverse != 0:
            imgs_sliced = slice_imgs([image_f(noise)], a.samples, a.modsize, norm_in, a.overscan, micro=micro)
            out_enc2 = model_clip.encode_image(imgs_sliced[-1])
            loss += a.diverse * torch.cosine_similarity(out_enc, out_enc2, dim=-1).mean()
            del out_enc2; torch.cuda.empty_cache()
        if a.in_img is not None and os.path.isfile(a.in_img): # input image
            loss +=  sign * 0.5 * torch.cosine_similarity(img_enc, out_enc, dim=-1).mean()
        if a.in_txt is not None: # input text
            loss +=  sign * torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()
        if a.in_txt0 is not None: # subtract text
            loss += -sign * torch.cosine_similarity(txt_enc0, out_enc, dim=-1).mean()
        if a.sync > 0 and a.in_img is not None and os.path.isfile(a.in_img): # image composition
            loss -= a.sync * ssim_loss(F.interpolate(img_out, ssim_size).float(), img_in)
        if a.in_txt2 is not None: # input text for micro details
            imgs_sliced = slice_imgs([img_out], a.samples, a.modsize, norm_in, a.overscan, micro=True)
            out_enc2 = model_clip.encode_image(imgs_sliced[-1])
            loss +=  sign * torch.cosine_similarity(txt_enc2, out_enc2, dim=-1).mean()
            del out_enc2; torch.cuda.empty_cache()
        if a.expand > 0:
            global prev_enc
            if i > 0:
                loss += a.expand * torch.cosine_similarity(out_enc, prev_enc, dim=-1).mean()
            prev_enc = out_enc.detach()

        del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()
        assert not isinstance(loss, int), ' Loss not defined, check the inputs'
        
        if a.prog is True:
            lr_cur = lr0 + (i / a.steps) * (lr1 - lr0)
            for g in optimizer.param_groups: 
                g['lr'] = lr_cur
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % a.fstep == 0:
            with torch.no_grad():
                img = image_f(contrast=a.contrast).cpu().numpy()[0]
            checkout(img, os.path.join(tempdir, '%04d.jpg' % (i // a.fstep)), verbose=a.verbose)
            pbar.upd()

    # Load CLIP models
    model_clip, _ = clip.load(a.model)
    if a.verbose is True: print(' using model', a.model)
    xmem = {'RN50':0.5, 'RN50x4':0.16, 'RN101':0.33}
    if 'RN' in a.model:
        a.samples = int(a.samples * xmem[a.model])
            
    if a.multilang is True:
        model_lang = SentenceTransformer('clip-ViT-B-32-multilingual-v1').cuda()

    def enc_text(txt):
        if a.multilang is True:
            emb = model_lang.encode([txt], convert_to_tensor=True, show_progress_bar=False)
        else:
            emb = model_clip.encode_text(clip.tokenize(txt).cuda())
        return emb.detach().clone()
    
    if a.diverse != 0:
        a.samples = int(a.samples * 0.5)
            
    norm_in = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    out_name = []
    if a.in_img is not None and os.path.isfile(a.in_img):
        if a.verbose is True: print(' ref image:', basename(a.in_img))
        img_in = torch.from_numpy(img_read(a.in_img)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
        img_in = img_in[:,:3,:,:] # fix rgb channels
        in_sliced = slice_imgs([img_in], a.samples, a.modsize, transform=norm_in, overscan=a.overscan)[0]
        img_enc = model_clip.encode_image(in_sliced).detach().clone()
        if a.sync > 0:
            ssim_loss = ssim.SSIM(window_size = 11)
            ssim_size = [s//8 for s in a.size]
            img_in = F.interpolate(img_in, ssim_size).float()
        else:
            del img_in
        del in_sliced; torch.cuda.empty_cache()
        out_name.append(basename(a.in_img).replace(' ', '_'))

    if a.in_txt is not None:
        if a.verbose is True: print(' ref text: ', basename(a.in_txt))
        if a.translate:
            translator = Translator()
            a.in_txt = translator.translate(a.in_txt, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt) 
        txt_enc = enc_text(a.in_txt)
        out_name.append(txt_clean(a.in_txt))

    if a.in_txt2 is not None:
        if a.verbose is True: print(' micro text:', basename(a.in_txt2))
        a.samples = int(a.samples * 0.75)
        if a.translate:
            translator = Translator()
            a.in_txt2 = translator.translate(a.in_txt2, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt2) 
        txt_enc2 = enc_text(a.in_txt2)
        out_name.append(txt_clean(a.in_txt2))

    if a.in_txt0 is not None:
        if a.verbose is True: print(' subtract text:', basename(a.in_txt0))
        a.samples = int(a.samples * 0.75)
        if a.translate:
            translator = Translator()
            a.in_txt0 = translator.translate(a.in_txt0, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt0) 
        txt_enc0 = enc_text(a.in_txt0)
        out_name.append('off-' + txt_clean(a.in_txt0))

    if a.multilang is True: del model_lang

    params, image_f = fft_image([1, 3, *a.size], resume=a.resume)
    image_f = to_valid_rgb(image_f, colors = a.colors)

    if a.prog is True:
        lr1 = a.lrate * 2
        lr0 = lr1 * 0.01
    else:
        lr0 = a.lrate
    optimizer = torch.optim.Adam(params, lr0)
    sign = 1. if a.invert is True else -1.

    if a.verbose is True: print(' samples:', a.samples)
    out_name = '-'.join(out_name)
    out_name += '-%s' % a.model if 'RN' in a.model.upper() else ''
    tempdir = os.path.join(a.out_dir, out_name)
    os.makedirs(tempdir, exist_ok=True)

    pbar = ProgressBar(a.steps // a.fstep)
    for i in range(a.steps):
        train(i)

    os.system('ffmpeg -v warning -y -i %s\%%04d.jpg "%s.mp4"' % (tempdir, os.path.join(a.out_dir, out_name)))
    shutil.copy(img_list(tempdir)[-1], os.path.join(a.out_dir, '%s-%d.jpg' % (out_name, a.steps)))
    if a.save_pt is True:
        torch.save(params, '%s.pt' % os.path.join(a.out_dir, out_name))

if __name__ == '__main__':
    main()
