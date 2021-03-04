# coding: 1251
import os
import argparse
import numpy as np
import cv2
import shutil
from googletrans import Translator, constants

import torch
import torchvision
import torch.nn.functional as F

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from clip_fft import to_valid_rgb, fft_image, slice_imgs, checkout
from utils import pad_up_to, basename, file_list, img_list, img_read
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_txt', default=None, help='Text file to process')
    parser.add_argument('--out_dir', default='_out')
    parser.add_argument('-s', '--size', default='1280-720', help='Output resolution')
    parser.add_argument('-l', '--length', default=180, type=int, help='Total length in sec')
    parser.add_argument('-r', '--resume', default=None, help='Path to saved params, to resume from')
    parser.add_argument('--fstep', default=1, type=int, help='Saving step')
    parser.add_argument('-tr', '--translate', action='store_true')
    parser.add_argument('--save_pt', action='store_true', help='Save FFT params for further use')
    parser.add_argument('--fps', default=25, type=int)
    parser.add_argument('-v', '--verbose', default=True, type=bool)
    # training
    parser.add_argument('--samples', default=200, type=int, help='Samples to evaluate')
    parser.add_argument('--lrate', default=0.05, type=float, help='Learning rate')
    parser.add_argument('-n', '--noise', default=0., type=float, help='Add noise to suppress accumulation')
    parser.add_argument('-o', '--overscan', default=True, help='Extra padding to add seamless tiling')
    parser.add_argument('-d', '--dual', action='store_true', help='Use both CLIP models at once')
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    return a

def main():
    a = get_args()

    # def train(i):
    def train(i, txt_enc, tempdir, pbar):
        loss = 0
        
        noise = a.noise * torch.randn(1, 1, *params[0].shape[2:4], 1).cuda() if a.noise > 0 else 0.
        img_out = image_f(noise)

        imgs_sliced = slice_imgs([img_out], a.samples, norm_in, a.overscan, micro=None)
        out_enc = model_vit.encode_image(imgs_sliced[-1])
        if a.dual is True: # use both clip models
            out_enc = torch.cat((out_enc, model_rn.encode_image(imgs_sliced[-1])), 1)
        loss -= 100*torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()
        del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % a.fstep == 0:
            with torch.no_grad():
                img = image_f().cpu().numpy()[0]
            checkout(img, os.path.join(tempdir, '%03d.jpg' % (i // a.fstep)), verbose=a.verbose)
            pbar.upd()

    # Load CLIP models
    model_vit, _ = clip.load('ViT-B/32')
    if a.dual is True:
        if a.verbose is True: print(' using dual-model optimization')
        model_rn, _ = clip.load('RN50', path='models/RN50.pt ')
        a.samples = a.samples // 2
            
    norm_in = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    params, image_f = fft_image([1, 3, *a.size], resume=a.resume)
    image_f = to_valid_rgb(image_f)

    optimizer = torch.optim.Adam(params, a.lrate) # pixel 1, fft 0.05

    with open(a.in_txt, 'r', encoding="utf-8") as f:
        texts = f.readlines()
        texts = [tt.strip() for tt in texts if len(tt.strip())>0 and tt[0] != '#']
    a.steps = int(a.length * a.fps * a.fstep / len(texts))
    workdir = os.path.join(a.out_dir, basename(a.in_txt))
    if a.verbose is True: 
        print(' total lines:', len(texts))
        print(' samples:', a.samples)

    def process(txt, num):
        if a.verbose is True: print(' ref text: ', txt)
        if a.translate:
            translator = Translator()
            txt = translator.translate(txt, dest='en').text
            if a.verbose is True: print(' translated to:', txt)
        tx = clip.tokenize(txt).cuda()
        txt_enc = model_vit.encode_text(tx).detach().clone()
        if a.dual is True:
            txt_enc = torch.cat((txt_enc, model_rn.encode_text(tx).detach().clone()), 1)
        out_name = '%02d-%s' % (num, txt.translate(str.maketrans(dict.fromkeys(list("\n',—|!?/:;\\"), ""))).replace(' ', '_').replace('"', ''))
        tempdir = os.path.join(workdir, out_name)
        os.makedirs(tempdir, exist_ok=True)
        
        pbar = ProgressBar(a.steps // a.fstep)
        for i in range(a.steps):
            train(i, txt_enc, tempdir, pbar)

        if a.save_pt is True:
            torch.save(params, '%s.pt' % os.path.join(workdir, out_name))
        shutil.copy(img_list(tempdir)[-1], os.path.join(workdir, '%s-%d.jpg' % (out_name, a.steps)))
        os.system('ffmpeg -v warning -y -i %s\%%03d.jpg "%s.mp4"' % (tempdir, os.path.join(workdir, out_name)))

    for i, txt in enumerate(texts):
        process(txt, i)

    vid_list = ['file ' + v.replace('\\', '/') for v in file_list(workdir, 'mp4')]
    with open('dir.txt', 'w') as ff:
        ff.write('\n'.join(vid_list))
    os.system('ffmpeg -y -v warning -f concat -i dir.txt -c:v copy %s.mp4' % os.path.join(a.out_dir, basename(a.in_txt)))
    os.remove('dir.txt')

if __name__ == '__main__':
    main()
