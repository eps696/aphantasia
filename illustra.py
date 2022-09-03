# coding: UTF-8
import os
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import random
import shutil

import torch
import torchvision
import torch.nn.functional as F

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from aphantasia.image import to_valid_rgb, fft_image
from aphantasia.utils import slice_imgs, derivat, checkout, basename, file_list, img_list, img_read, txt_clean, old_torch, save_cfg, sim_func, aesthetic_model
from aphantasia import transforms
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from aphantasia.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from aphantasia.progress_bar import ProgressBar

clip_models = ['ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'ViT-L/14@336px', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',  '--size',    default='1280-720', help='Output resolution')
    parser.add_argument('-t',  '--in_txt',  default=None, help='input text or file - main topic')
    parser.add_argument('-t2', '--in_txt2', default=None, help='input text or file - style')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images')
    parser.add_argument('-r',  '--resume',  default=None, help='Resume from saved params')
    parser.add_argument(       '--out_dir', default='_out/fft')
    parser.add_argument(     '--save_step', default=1, type=int, help='Save every this step')
    parser.add_argument('-tr', '--translate', action='store_true', help='Translate with Google Translate')
    parser.add_argument('-v',  '--verbose',    dest='verbose', action='store_true')
    parser.add_argument('-nv', '--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    # training
    parser.add_argument('-m',  '--model',   default='ViT-B/32', choices=clip_models, help='Select CLIP model to use')
    parser.add_argument(       '--steps',   default=150, type=int, help='Iterations per input')
    parser.add_argument(       '--samples', default=200, type=int, help='Samples to evaluate')
    parser.add_argument('-lr', '--lrate',   default=0.05, type=float, help='Learning rate')
    parser.add_argument('-dm', '--dualmod', default=None, type=int, help='Every this step use another CLIP ViT model')
    # tweaks
    parser.add_argument('-opt', '--optimr', default='adam', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('-a',  '--align',   default='uniform', choices=['central', 'uniform', 'overscan', 'overmax'], help='Sampling distribution')
    parser.add_argument('-tf', '--transform', default='fast', choices=['none', 'custom', 'fast', 'elastic'], help='augmenting transforms')
    parser.add_argument(       '--aest',    default=1., type=float)
    parser.add_argument(       '--contrast', default=1.1, type=float)
    parser.add_argument(       '--colors',  default=1.8, type=float)
    parser.add_argument('-d',  '--decay',   default=1.5, type=float)
    parser.add_argument('-sh', '--sharp',   default=0, type=float)
    parser.add_argument('-mc', '--macro',   default=0.4, type=float, help='Endorse macro forms 0..1 ')
    parser.add_argument('-e',  '--enforce', default=0, type=float, help='Enhance consistency, boosts training')
    parser.add_argument('-n',  '--noise',   default=0, type=float, help='Add noise to decrease accumulation')
    parser.add_argument(       '--sim',     default='mix', help='Similarity function (dot/angular/spherical/mixed; None = cossim)')
    parser.add_argument(       '--loop',    action='store_true', help='Loop inputs [or keep the last one]')
    parser.add_argument(       '--save_pt', action='store_true', help='save fft snapshots to pt file')
    # multi input
    parser.add_argument('-l',  '--length',  default=None, type=int, help='Override total length in sec')
    parser.add_argument(       '--lsteps',  default=25, type=int, help='Frames per step')
    parser.add_argument(       '--fps',     default=25, type=int)
    parser.add_argument(       '--keep',    default=1.5, type=float, help='Accumulate imagery: 0 random, 1+ ~prev')
    parser.add_argument(       '--separate', action='store_true', help='process inputs separately')
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    if not a.separate: a.save_pt = True
    if a.dualmod is not None: 
        a.model = 'ViT-B/32'
        a.sim = 'cossim'

    return a

a = get_args()

if a.translate is True:
    try:
        from googletrans import Translator
    except ImportError as e:
        print('\n Install googletrans module to enable translation!'); exit()

def main():
    bx = 1.

    model_clip, _ = clip.load(a.model, jit=old_torch())
    try:
        a.modsize = model_clip.visual.input_resolution
    except:
        a.modsize = 288 if a.model == 'RN50x4' else 384 if a.model == 'RN50x16' else 448 if a.model == 'RN50x64' else 336 if '336' in a.model else 224
    model_clip = model_clip.eval().cuda()
    xmem = {'ViT-B/16':0.25, 'ViT-L/14':0.04, 'RN50':0.5, 'RN50x4':0.16, 'RN50x16':0.06, 'RN50x64':0.01, 'RN101':0.33}
    if a.model in xmem.keys():
        bx *= xmem[a.model]

    if a.dualmod is not None:
        model_clip2, _ = clip.load('ViT-B/16', jit=old_torch())
        bx *= 0.23 # second is vit-16
        dualmod_nums = list(range(a.steps))[a.dualmod::a.dualmod]
        print(' dual model every %d step' % a.dualmod)

    if a.aest != 0 and a.model in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        aest = aesthetic_model(a.model).cuda()
        if a.dualmod is not None:
            aest2 = aesthetic_model('ViT-B/16').cuda()
    
    if 'elastic' in a.transform:
        trform_f = transforms.transforms_elastic
    elif 'custom' in a.transform:
        trform_f = transforms.transforms_custom
    elif 'fast' in a.transform:
        trform_f = transforms.transforms_fast
    else:
        trform_f = transforms.normalize()
        bx *= 1.05
    bx *= 0.95
    if a.enforce != 0:
        bx *= 0.5
    a.samples = int(bx * a.samples)

    if a.translate:
        translator = Translator()

    def enc_text(txt, model_clip=model_clip):
        if txt is None or len(txt)==0: return None
        embs = []
        for subtxt in txt.split('|'):
            if ':' in subtxt:
                [subtxt, wt] = subtxt.split(':')
                wt = float(wt)
            else: wt = 1.
            emb = model_clip.encode_text(clip.tokenize(subtxt).cuda()[:77])
            # emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append([emb.detach().clone(), wt])
        return embs

    def enc_image(img, model_clip=model_clip):
        emb = model_clip.encode_image(img)
        # emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb
    
    def proc_image(img_file, model_clip=model_clip):
        img_t = torch.from_numpy(img_read(img_file)/255.).unsqueeze(0).permute(0,3,1,2).cuda()[:,:3,:,:]
        in_sliced = slice_imgs([img_t], a.samples, a.modsize, transforms.normalize(), a.align)[0]
        emb = enc_image(in_sliced, model_clip)
        return emb.detach().clone()

    def pick_(list_, num_):
        cnt = len(list_)
        if cnt == 0: return None
        num = num_ % cnt if a.loop is True else min(num_, cnt-1)
        return list_[num]

    def read_text(in_txt):
        if os.path.isfile(in_txt):
            with open(in_txt, 'r', encoding="utf-8") as f:
                lines = f.read().splitlines()
            texts = []
            for tt in lines:
                if len(tt.strip()) == 0: texts.append('')
                elif tt.strip()[0] != '#': texts.append(tt.strip())
        else:
            texts = [in_txt]
        return texts
    
    # Encode inputs
    count = 0
    texts = []
    styles = []
    img_paths = []

    if a.in_img is not None and os.path.exists(a.in_img):
        if a.verbose is True: print(' ref image:', basename(a.in_img))
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
    img_encs = [proc_image(image) for image in img_paths]
    if a.dualmod is not None:
        img_encs2 = [proc_image(image, model_clip2) for image in img_paths]
    count = max(count, len(img_encs))

    if a.in_txt is not None:
        if a.verbose is True: print(' topic:', a.in_txt)
        texts = read_text(a.in_txt)
    if a.translate:
        texts = [translator.translate(txt, dest='en').text for txt in texts]
        # if a.verbose is True: print(' translated to:', texts)
    txt_encs = [enc_text(txt) for txt in texts] 
    if a.dualmod is not None:
        txt_encs2 = [enc_text(txt, model_clip2) for txt in texts]
    count = max(count, len(txt_encs))

    if a.in_txt2 is not None:
        if a.verbose is True: print(' style:', a.in_txt2)
        styles = read_text(a.in_txt2)
    if a.translate is True:
        styles = [tr.text for tr in translator.translate(styles)]
        # if a.verbose is True: print(' translated to:', styles)
    styl_encs = [enc_text(style) for style in styles]
    if a.dualmod is not None:
        styl_encs2 = [enc_text(style, model_clip2) for style in styles]
    count = max(count, len(styl_encs))
       
    assert count > 0, "No inputs found!"

    if a.verbose is True: print(' samples:', a.samples)
    sfx = ''
    if a.dualmod is None:        sfx += '-%s' % a.model.replace('/','').replace('-','') 
    if a.enforce != 0:           sfx += '-e%.2g' % a.enforce
    # if a.noise > 0:              sfx += '-n%.2g' % a.noise
    # if a.aest != 0:              sfx += '-ae%.2g' % a.aest
    
    def train(num, i):
        loss = 0
        noise = a.noise * (torch.rand(1, 1, *params[0].shape[2:4], 1)-0.5).cuda() if a.noise > 0 else None
        img_out = image_f(noise)
        img_sliced = slice_imgs([img_out], a.samples, a.modsize, trform_f, a.align, a.macro)[0]
        
        if a.in_txt is not None:
            txt_enc   = pick_(txt_encs2, num)  if a.dualmod is not None and i in dualmod_nums else pick_(txt_encs, num)
        if a.in_txt2 is not None:
            style_enc = pick_(styl_encs2, num) if a.dualmod is not None and i in dualmod_nums else pick_(styl_encs, num)
        if a.in_img is not None and os.path.isfile(a.in_img):
            img_enc   = pick_(img_encs2, num)  if a.dualmod is not None and i in dualmod_nums else pick_(img_encs, num)
        model_clip_   = model_clip2   if a.dualmod is not None and i in dualmod_nums else model_clip
        if a.aest != 0:
            aest_     = aest2         if a.dualmod is not None and i in dualmod_nums else aest

        out_enc = model_clip_.encode_image(img_sliced)
        if a.aest != 0 and aest_ is not None:
            loss -= 0.001 * a.aest * aest_(out_enc).mean()
        if a.in_txt is not None and txt_enc is not None: # input text - main topic
            for enc, wt in txt_enc:
                loss -= wt * sim_func(enc, out_enc, a.sim)
        if a.in_txt2 is not None and style_enc is not None: # input text - style
            for enc, wt in style_enc:
                loss -= wt * sim_func(enc, out_enc, a.sim)
        if a.in_img is not None and img_enc is not None: # input image
            loss -= sim_func(img_enc[:len(out_enc)], out_enc, a.sim)
        if a.sharp != 0: # scharr|sobel|naiv
            loss -= a.sharp * derivat(img_out, mode='naiv')
        if a.enforce != 0:
            img_sliced = slice_imgs([image_f(noise)], a.samples, a.modsize, trform_f, a.align, a.macro)[0]
            out_enc2 = model_clip_.encode_image(img_sliced)
            loss -= a.enforce * sim_func(out_enc, out_enc2, a.sim)
            del out_enc2 # torch.cuda.empty_cache()

        del img_out, img_sliced, out_enc
        assert not isinstance(loss, int), ' Loss not defined, check inputs'
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % a.save_step == 0:
            with torch.no_grad():
                img = image_f(contrast=a.contrast).cpu().numpy()[0]
            checkout(img, os.path.join(tempdir, '%04d.jpg' % (i // a.save_step)), verbose=a.verbose)
            pbar.upd()
            del img


    try:
        for num in range(count):
            shape = [1, 3, *a.size]
            global params

            if num == 0 or a.separate is True:
                resume_cur = a.resume
            else:
                opt_state = optimizer.state_dict()
                param_ = params[0].detach()
                resume_cur = [a.keep * param_ / (param_.max() - param_.min())]

            params, image_f, sz = fft_image(shape, 0.08, a.decay, resume_cur)
            if sz is not None: a.size = sz
            image_f = to_valid_rgb(image_f, colors = a.colors)

            if a.optimr.lower() == 'adamw':
                optimizer = torch.optim.AdamW(params, a.lrate, weight_decay=0.01, betas=(.0,.999), amsgrad=True)
            else:
                optimizer = torch.optim.Adam(params, a.lrate, betas=(.0, .999))
            if num > 0 and not a.separate: optimizer.load_state_dict(opt_state)

            out_names = []
            if a.resume  is not None and num == 0: out_names += [basename(a.resume)[:12]]
            if a.in_txt  is not None: out_names += [txt_clean(pick_(texts, num))[:32]]
            if a.in_txt2 is not None: out_names += [txt_clean(pick_(styles, num))[:32]]
            out_name = '-'.join(out_names) + sfx
            if count > 1: out_name = '%04d-' % (num+1) + out_name
            print(out_name)
            workdir = a.out_dir
            tempdir = os.path.join(workdir, out_name)
            os.makedirs(tempdir, exist_ok=True)
            if num == 0: save_cfg(a, workdir, out_name + '.txt')

            pbar = ProgressBar(a.steps // a.save_step)
            for i in range(a.steps):
                train(num, i)

            file_out = os.path.join(workdir, '%s-%d.jpg' % (out_name, a.steps))
            shutil.copy(img_list(tempdir)[-1], file_out)
            os.system('ffmpeg -v warning -y -i %s\%%04d.jpg "%s.mp4"' % (tempdir, os.path.join(workdir, out_name)))
            if a.save_pt is True:
                torch.save(params[0], '%s.pt' % os.path.join(workdir, out_name))

    except KeyboardInterrupt:
        exit()

    if not a.separate:
        vsteps = a.lsteps if a.length is None else int(a.length * a.fps / count)
        tempdir = os.path.join(workdir, '_final')
        os.makedirs(tempdir, exist_ok=True)
        
        def read_pt(file):
            return torch.load(file).cuda()

        if a.verbose is True: print(' rendering complete piece')
        ptfiles = file_list(workdir, 'pt')
        pbar = ProgressBar(vsteps * len(ptfiles))
        for px in range(len(ptfiles)):
            params1 = read_pt(ptfiles[px])
            params2 = read_pt(ptfiles[(px+1) % len(ptfiles)])

            params, image_f, sz_ = fft_image([1, 3, *a.size], resume=params1, sd=1., decay_power=a.decay)
            image_f = to_valid_rgb(image_f, colors = a.colors)

            for i in range(vsteps):
                with torch.no_grad():
                    x = i/vsteps # math.sin(1.5708 * i/vsteps)
                    img = image_f((params2 - params1) * x, contrast=a.contrast).cpu().numpy()[0]
                checkout(img, os.path.join(tempdir, '%05d.jpg' % (px * vsteps + i)), verbose=a.verbose)
                pbar.upd()

        os.system('ffmpeg -v warning -y -i %s/\%%05d.jpg "%s.mp4"' % (tempdir, os.path.join(a.out_dir, basename(a.in_txt))))


if __name__ == '__main__':
    main()
