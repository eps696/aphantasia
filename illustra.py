import os
import argparse
import math
import numpy as np
import shutil
from imageio import imsave

try:
    from googletrans import Translator
    googletrans_ok = True
except ImportError as e:
    googletrans_ok = False

import torch
import torchvision
import torch.nn.functional as F

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sentence_transformers import SentenceTransformer

from aphantasia.image import to_valid_rgb, fft_image
from aphantasia.utils import slice_imgs, derivat, checkout, cvshow, pad_up_to, basename, file_list, img_list, img_read, txt_clean, plot_text, old_torch
from aphantasia import transforms
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from aphantasia.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from aphantasia.progress_bar import ProgressBar

clip_models = ['ViT-B/16', 'ViT-B/32', 'RN101', 'RN50x16', 'RN50x4', 'RN50']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--in_txt',  default=None, help='Text file to process')
    parser.add_argument('-t2', '--in_txt2', default=None, help='input text - style')
    parser.add_argument('-t0', '--in_txt0', default=None, help='input text to subtract')
    parser.add_argument(       '--out_dir', default='_out')
    parser.add_argument('-s',  '--size',    default='1280-720', help='Output resolution')
    parser.add_argument('-r',  '--resume',  default=None, help='Path to saved FFT snapshots, to resume from')
    parser.add_argument('-l',  '--length',  default=180, type=int, help='Total length in sec')
    parser.add_argument(       '--fstep',   default=1, type=int, help='Saving step')
    parser.add_argument('-tr', '--translate', action='store_true', help='Translate text with Google Translate')
    parser.add_argument('-ml', '--multilang', action='store_true', help='Use SBERT multilanguage model for text')
    parser.add_argument(       '--save_pt', action='store_true', help='Save FFT snapshots for further use')
    parser.add_argument(       '--fps',     default=25, type=int)
    parser.add_argument('-v',  '--verbose',    dest='verbose', action='store_true')
    parser.add_argument('-nv', '--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    # training
    parser.add_argument('-m',  '--model',   default='ViT-B/32', choices=clip_models, help='Select CLIP model to use')
    parser.add_argument(       '--steps',   default=200, type=int, help='Total iterations')
    parser.add_argument(       '--samples', default=200, type=int, help='Samples to evaluate')
    parser.add_argument('-lr', '--lrate',   default=0.05, type=float, help='Learning rate')
    parser.add_argument('-p',  '--prog',    action='store_true', help='Enable progressive lrate growth (up to double a.lrate)')
    # tweaks
    parser.add_argument('-a',  '--align',   default='uniform', choices=['central', 'uniform', 'overscan'], help='Sampling distribution')
    parser.add_argument('-tf', '--transform', default='fast', choices=['none', 'fast', 'custom', 'elastic'], help='augmenting transforms')
    parser.add_argument(       '--keep',    default=0, type=float, help='Accumulate imagery: 0 = random, 1 = prev ema')
    parser.add_argument(       '--contrast', default=0.9, type=float)
    parser.add_argument(       '--colors',  default=1.5, type=float)
    parser.add_argument(       '--decay',   default=1.5, type=float)
    parser.add_argument('-sh', '--sharp',   default=0.3, type=float)
    parser.add_argument('-mm', '--macro',   default=0.4, type=float, help='Endorse macro forms 0..1 ')
    parser.add_argument('-e',  '--enforce', default=0, type=float, help='Enforce details (by boosting similarity between two parallel samples)')
    parser.add_argument('-x',  '--expand',  default=0, type=float, help='Boosts diversity (by enforcing difference between prev/next samples)')
    parser.add_argument('-n',  '--noise',   default=0.2, type=float, help='Add noise to suppress accumulation')
    parser.add_argument('-nt', '--notext',  default=0, type=float, help='Subtract typed text as image (avoiding graffiti?), [0..1]') # 0.15
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    if a.multilang is True: a.model = 'ViT-B/32' # sbert model is trained with ViT
    if a.translate is True and googletrans_ok is not True: 
        print('\n Install googletrans module to enable translation!'); exit()
    
    return a

def ema(base, next, step):
    scale_ma = 1. / (step + 1)
    return next * scale_ma + base * (1.- scale_ma)

def load_params(file):
    if not os.path.isfile(file):
        print(' Snapshot not found:', file); exit()
    params = torch.load(file)
    if isinstance(params, list): params = params[0]
    return params.detach().clone()

def main():
    a = get_args()

    # Load CLIP models
    model_clip, _ = clip.load(a.model, jit=old_torch())
    try:
        a.modsize = model_clip.visual.input_resolution 
    except:
        a.modsize = 288 if a.model == 'RN50x4' else 384 if a.model == 'RN50x16' else 224
    if a.verbose is True: print(' using model', a.model)
    xmem = {'ViT-B/16':0.25, 'RN50':0.5, 'RN50x4':0.16, 'RN50x16':0.06, 'RN101':0.33}
    if a.model in xmem.keys():
        a.samples = int(a.samples * xmem[a.model])
    workdir = os.path.join(a.out_dir, basename(a.in_txt))
    workdir += '-%s' % a.model if 'RN' in a.model.upper() else ''
    os.makedirs(workdir, exist_ok=True)

    def enc_text(txt):
        if a.multilang is True:
            model_lang = SentenceTransformer('clip-ViT-B-32-multilingual-v1').cuda()
            emb = model_lang.encode([txt], convert_to_tensor=True, show_progress_bar=False)
            del model_lang
        else:
            emb = model_clip.encode_text(clip.tokenize(txt).cuda())
        return emb.detach().clone()
    
    if a.enforce != 0:
        a.samples = int(a.samples * 0.5)
            
    if 'elastic' in a.transform:
        trform_f = transforms.transforms_elastic
        a.samples = int(a.samples * 0.95)
    elif 'custom' in a.transform:
        trform_f = transforms.transforms_custom
        a.samples = int(a.samples * 0.95)
    elif 'fast' in a.transform:
        trform_f = transforms.transforms_fast
        a.samples = int(a.samples * 0.95)
    else:
        trform_f = transforms.normalize()

    if a.in_txt2 is not None:
        if a.verbose is True: print(' style:', basename(a.in_txt2))
        # a.samples = int(a.samples * 0.75)
        if a.translate:
            translator = Translator()
            a.in_txt2 = translator.translate(a.in_txt2, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt2)
        txt_enc2 = enc_text(a.in_txt2)

    if a.in_txt0 is not None:
        if a.verbose is True: print(' subtract text:', basename(a.in_txt0))
        if a.translate:
            translator = Translator()
            a.in_txt0 = translator.translate(a.in_txt0, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt0) 
        txt_enc0 = enc_text(a.in_txt0)

    # make init
    global params_start, params_ema
    params_shape = [1, 3, a.size[0], a.size[1]//2+1, 2]
    params_start = torch.randn(*params_shape).cuda() # random init
    params_ema = 0.
    if a.resume is not None and os.path.isfile(a.resume):
        if a.verbose is True: print(' resuming from', a.resume)
        params_start = load_params(a.resume).cuda()
        if a.keep > 0:
            params_ema = params_start[0].detach().clone()
    else:
        a.resume = 'init.pt'

    torch.save(params_start, 'init.pt') # final init
    shutil.copy(a.resume, os.path.join(workdir, '000-%s.pt' % basename(a.resume)))
    
    prev_enc = 0
    def process(txt, num):

        sd = 0.01
        if a.keep > 0: sd = a.keep + (1-a.keep) * sd
        params, image_f, _ = fft_image([1, 3, *a.size], resume='init.pt', sd=sd, decay_power=a.decay)
        image_f = to_valid_rgb(image_f, colors = a.colors)

        if a.prog is True:
            lr1 = a.lrate * 2
            lr0 = a.lrate * 0.1
        else:
            lr0 = a.lrate
        optimizer = torch.optim.AdamW(params, lr0, weight_decay=0.01, amsgrad=True)
    
        if a.verbose is True: print(' topic: ', txt)
        if a.translate:
            translator = Translator()
            txt = translator.translate(txt, dest='en').text
            if a.verbose is True: print(' translated to:', txt)
        txt_enc = enc_text(txt)
        if a.notext > 0:
            txt_plot = torch.from_numpy(plot_text(txt, a.modsize)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
            txt_plot_enc = model_clip.encode_image(txt_plot).detach().clone()
        else: txt_plot_enc = None

        out_name = '%03d-%s' % (num+1, txt_clean(txt))
        out_name += '-%s' % a.model if 'RN' in a.model.upper() else ''
        tempdir = os.path.join(workdir, out_name)
        os.makedirs(tempdir, exist_ok=True)
        
        pbar = ProgressBar(a.steps // a.fstep)
        for i in range(a.steps):
            loss = 0
            noise = a.noise * torch.randn(1, 1, *params[0].shape[2:4], 1).cuda() if a.noise > 0 else None
            img_out = image_f(noise)
            img_sliced = slice_imgs([img_out], a.samples, a.modsize, trform_f, a.align, macro=a.macro)[0]
            out_enc = model_clip.encode_image(img_sliced)

            loss -= torch.cosine_similarity(txt_enc, out_enc, dim=-1).mean()
            if a.in_txt2 is not None: # input text - style
                loss -= 0.5 * torch.cosine_similarity(txt_enc2, out_enc, dim=-1).mean()
            if a.in_txt0 is not None: # subtract text
                loss += 0.5 * torch.cosine_similarity(txt_enc0, out_enc, dim=-1).mean()
            if a.notext > 0:
                loss += a.notext * torch.cosine_similarity(txt_plot_enc, out_enc, dim=-1).mean()
            if a.sharp != 0: # mode = scharr|sobel|default
                loss -= a.sharp * derivat(img_out, mode='sobel')
                # loss -= a.sharp * derivat(img_sliced, mode='scharr')
            if a.enforce != 0:
                img_sliced = slice_imgs([image_f(noise)], a.samples, a.modsize, trform_f, a.align, macro=a.macro)[0]
                out_enc2 = model_clip.encode_image(img_sliced)
                loss -= a.enforce * torch.cosine_similarity(out_enc, out_enc2, dim=-1).mean()
                del out_enc2; torch.cuda.empty_cache()
            if a.expand > 0:
                global prev_enc
                if i > 0:
                    loss += a.expand * torch.cosine_similarity(out_enc, prev_enc, dim=-1).mean()
                prev_enc = out_enc.detach().clone()
            del img_out, img_sliced, out_enc; torch.cuda.empty_cache()

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
                if a.sharp != 0:
                    img = img ** (1 + a.sharp/2.) # empirical tone mapping
                checkout(img, os.path.join(tempdir, '%04d.jpg' % (i // a.fstep)), verbose=a.verbose)
                pbar.upd()
                del img

        if a.keep > 0:
            global params_start, params_ema
            params_ema = ema(params_ema, params[0].detach().clone(), num+1)
            torch.save((1-a.keep) * params_start + a.keep * params_ema, 'init.pt')
        
        torch.save(params[0], '%s.pt' % os.path.join(workdir, out_name))
        shutil.copy(img_list(tempdir)[-1], os.path.join(workdir, '%s-%d.jpg' % (out_name, a.steps)))
        os.system('ffmpeg -v warning -y -i %s\%%04d.jpg "%s.mp4"' % (tempdir, os.path.join(workdir, out_name)))

    with open(a.in_txt, 'r', encoding="utf-8") as f:
        texts = f.readlines()
        texts = [tt.strip() for tt in texts if len(tt.strip()) > 0 and tt[0] != '#']
    if a.verbose is True: 
        print(' total lines:', len(texts))
        print(' samples:', a.samples)

    for i, txt in enumerate(texts):
        process(txt, i)

    vsteps = int(a.length * 25 / len(texts)) # 25 fps
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

        params, image_f, _ = fft_image([1, 3, *a.size], resume=params1, sd=1., decay_power=a.decay)
        image_f = to_valid_rgb(image_f, colors = a.colors)

        for i in range(vsteps):
            with torch.no_grad():
                img = image_f((params2 - params1) * math.sin(1.5708 * i/vsteps)**2)[0].permute(1,2,0)
                img = torch.clip(img*255, 0, 255).cpu().numpy().astype(np.uint8)
            imsave(os.path.join(tempdir, '%05d.jpg' % (px * vsteps + i)), img)
            if a.verbose is True: cvshow(img)
            pbar.upd()

    os.system('ffmpeg -v warning -y -i %s/\%%05d.jpg "%s.mp4"' % (tempdir, os.path.join(a.out_dir, basename(a.in_txt))))
    if a.keep > 0: os.remove('init.pt')


if __name__ == '__main__':
    main()
