import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
from imageio import imread, imsave
import shutil

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
import lpips

from aphantasia.image import to_valid_rgb, fft_image, dwt_image
from aphantasia.utils import slice_imgs, derivat, sim_func, aesthetic_model, basename, img_list, img_read, plot_text, txt_clean, checkout, old_torch
from aphantasia import transforms
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from aphantasia.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from aphantasia.progress_bar import ProgressBar

clip_models = ['ViT-B/16', 'ViT-B/32', 'RN101', 'RN50x16', 'RN50x4', 'RN50']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--in_img',  default=None, help='input image')
    parser.add_argument('-t',  '--in_txt',  default=None, help='input text')
    parser.add_argument('-t2', '--in_txt2', default=None, help='input text - style')
    parser.add_argument('-w2', '--weight2', default=0.5, type=float, help='weight for style')
    parser.add_argument('-t0', '--in_txt0', default=None, help='input text to subtract')
    parser.add_argument(       '--out_dir', default='_out')
    parser.add_argument('-s',  '--size',    default='1280-720', help='Output resolution')
    parser.add_argument('-r',  '--resume',  default=None, help='Path to saved FFT snapshots, to resume from')
    parser.add_argument('-ops', '--opt_step', default=1, type=int, help='How many optimizing steps per save step')
    parser.add_argument('-tr', '--translate', action='store_true', help='Translate text with Google Translate')
    parser.add_argument('-ml', '--multilang', action='store_true', help='Use SBERT multilanguage model for text')
    parser.add_argument(       '--save_pt', action='store_true', help='Save FFT snapshots for further use')
    parser.add_argument('-v',  '--verbose',    dest='verbose', action='store_true')
    parser.add_argument('-nv', '--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    # training
    parser.add_argument('-m',  '--model',   default='ViT-B/32', choices=clip_models, help='Select CLIP model to use')
    parser.add_argument(       '--steps',   default=200, type=int, help='Total iterations')
    parser.add_argument(       '--samples', default=200, type=int, help='Samples to evaluate')
    parser.add_argument(       '--lrate',   default=0.05, type=float, help='Learning rate')
    parser.add_argument('-p',  '--prog',    action='store_true', help='Enable progressive lrate growth (up to double a.lrate)')
    parser.add_argument('-dm', '--dualmod', default=None, type=int, help='Every this step use another CLIP ViT model')
    # wavelet
    parser.add_argument(       '--dwt',     action='store_true', help='Use DWT instead of FFT')
    parser.add_argument('-w',  '--wave',    default='coif2', help='wavelets: db[1..], coif[1..], haar, dmey')
    # tweaks
    parser.add_argument('-a',  '--align',   default='uniform', choices=['central', 'uniform', 'overscan', 'overmax'], help='Sampling distribution')
    parser.add_argument('-tf', '--transform', default='fast', choices=['none', 'fast', 'custom', 'elastic'], help='augmenting transforms')
    parser.add_argument('-opt', '--optimizer', default='adam', choices=['adam', 'adamw', 'adam_custom', 'adamw_custom'], help='Optimizer')
    parser.add_argument(       '--contrast', default=1.1, type=float)
    parser.add_argument(       '--colors',  default=1.8, type=float)
    parser.add_argument(       '--decay',   default=1.5, type=float)
    parser.add_argument('-sh', '--sharp',   default=0., type=float)
    parser.add_argument('-mm', '--macro',   default=0.4, type=float, help='Endorse macro forms 0..1 ')
    parser.add_argument(       '--aest',    default=0., type=float, help='Enhance aesthetics')
    parser.add_argument('-e',  '--enforce', default=0, type=float, help='Enforce details (by boosting similarity between two parallel samples)')
    parser.add_argument('-x',  '--expand',  default=0, type=float, help='Boosts diversity (by enforcing difference between prev/next samples)')
    parser.add_argument('-n',  '--noise',   default=0, type=float, help='Add noise to suppress accumulation') # < 0.05 ?
    parser.add_argument('-nt', '--notext',  default=0, type=float, help='Subtract typed text as image (avoiding graffiti?), [0..1]')
    parser.add_argument('-c',  '--sync',    default=0, type=float, help='Sync output to input image')
    parser.add_argument(       '--invert',  action='store_true', help='Invert criteria')
    parser.add_argument(       '--sim',     default='mix', help='Similarity function (dot/angular/spherical/mixed; None = cossim)')
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    if (a.in_img is not None and a.sync != 0) or a.resume is not None: a.align = 'overscan'
    if a.multilang is True: a.model = 'ViT-B/32' # sbert model is trained with ViT
    if a.translate is True and googletrans_ok is not True: 
        print('\n Install googletrans module to enable translation!'); exit()
    if a.dualmod is not None: 
        a.model = 'ViT-B/32'
        a.sim = 'cossim'
    
    return a

def main():
    a = get_args()

    shape = [1, 3, *a.size]
    if a.dwt is True:
        params, image_f, sz = dwt_image(shape, a.wave, 0.3, a.colors, a.resume)
    else:
        params, image_f, sz = fft_image(shape, 0.07, a.decay, a.resume)
    if sz is not None: a.size = sz
    image_f = to_valid_rgb(image_f, colors = a.colors)

    if a.prog is True:
        lr1 = a.lrate * 2
        lr0 = lr1 * 0.01
    else:
        lr0 = a.lrate
    if a.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, lr0, weight_decay=0.01)
    elif a.optimizer.lower() == 'adamw_custom':
        optimizer = torch.optim.AdamW(params, lr0, weight_decay=0.01, betas=(.0,.999), amsgrad=True)
    elif a.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr0)
    else: # adam_custom
        optimizer = torch.optim.Adam(params, lr0, betas=(.0,.999))
    sign = 1. if a.invert is True else -1.

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
            
    if a.multilang is True:
        model_lang = SentenceTransformer('clip-ViT-B-32-multilingual-v1').cuda()

    if a.dualmod is not None: # second is vit-16
        model_clip2, _ = clip.load('ViT-B/16', jit=old_torch())
        a.samples = int(a.samples * 0.23)
        dualmod_nums = list(range(a.steps))[a.dualmod::a.dualmod]
        print(' dual model every %d step' % a.dualmod)

    if a.aest != 0 and a.model in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        aest = aesthetic_model(a.model).cuda()
        if a.dualmod is not None:
            aest2 = aesthetic_model('ViT-B/16').cuda()
    
    def enc_text(txt, model_clip=model_clip):
        if a.multilang is True:
            emb = model_lang.encode([txt], convert_to_tensor=True, show_progress_bar=False)
        else:
            emb = model_clip.encode_text(clip.tokenize(txt).cuda())
        return emb.detach().clone()
    
    if a.enforce != 0:
        a.samples = int(a.samples * 0.5)
    if a.sync > 0:
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

    out_name = []
    if a.in_txt is not None:
        if a.verbose is True: print(' topic text: ', a.in_txt)
        if a.translate:
            translator = Translator()
            a.in_txt = translator.translate(a.in_txt, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt) 
        txt_enc = enc_text(a.in_txt)
        out_name.append(txt_clean(a.in_txt).lower()[:40])
        if a.dualmod is not None:
            txt_enc2 = enc_text(a.in_txt, model_clip2)
        if a.notext > 0:
            txt_plot = torch.from_numpy(plot_text(a.in_txt, a.modsize)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
            txt_plot_enc = model_clip.encode_image(txt_plot).detach().clone()
            if a.dualmod is not None:
                txt_plot_enc2 = model_clip2.encode_image(txt_plot).detach().clone()

    if a.in_txt2 is not None:
        if a.verbose is True: print(' style text:', a.in_txt2)
        a.samples = int(a.samples * 0.75)
        if a.translate:
            translator = Translator()
            a.in_txt2 = translator.translate(a.in_txt2, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt2) 
        style_enc = enc_text(a.in_txt2)
        out_name.append(txt_clean(a.in_txt2).lower()[:40])
        if a.dualmod is not None:
            style_enc2 = enc_text(a.in_txt2, model_clip2)

    if a.in_txt0 is not None:
        if a.verbose is True: print(' subtract text:', a.in_txt0)
        a.samples = int(a.samples * 0.75)
        if a.translate:
            translator = Translator()
            a.in_txt0 = translator.translate(a.in_txt0, dest='en').text
            if a.verbose is True: print(' translated to:', a.in_txt0) 
        not_enc = enc_text(a.in_txt0)
        out_name.append('off-' + txt_clean(a.in_txt0).lower()[:40])
        if a.dualmod is not None:
            not_enc2 = enc_text(a.in_txt0, model_clip2)

    if a.multilang is True: del model_lang

    if a.in_img is not None and os.path.isfile(a.in_img):
        if a.verbose is True: print(' ref image:', basename(a.in_img))
        img_in = torch.from_numpy(img_read(a.in_img)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
        img_in = img_in[:,:3,:,:] # fix rgb channels
        in_sliced = slice_imgs([img_in], a.samples, a.modsize, transforms.normalize(), a.align)[0]
        img_enc = model_clip.encode_image(in_sliced).detach().clone()
        if a.dualmod is not None:
            img_enc2 = model_clip2.encode_image(in_sliced).detach().clone()
        if a.sync > 0:
            sim_loss = lpips.LPIPS(net='vgg', verbose=False).cuda()
            sim_size = [s//2 for s in a.size]
            img_in = F.interpolate(img_in, sim_size, mode='bicubic', align_corners=True).float()
        else:
            del img_in
        del in_sliced; torch.cuda.empty_cache()
        out_name.append(basename(a.in_img).replace(' ', '_'))

    if a.verbose is True: print(' samples:', a.samples)
    out_name = '-'.join(out_name)
    out_name += '-%s' % a.model.replace('/','').replace('-','') if a.dualmod is None else '-dm%d' % a.dualmod
    tempdir = os.path.join(a.out_dir, out_name)
    os.makedirs(tempdir, exist_ok=True)

    prev_enc = 0
    def train(i):
        loss = 0
        
        noise = a.noise * torch.rand(1, 1, *params[0].shape[2:4], 1).cuda() if a.noise > 0 else None
        img_out = image_f(noise)
        img_sliced = slice_imgs([img_out], a.samples, a.modsize, trform_f, a.align, a.macro)[0]

        if a.in_txt is not None: # input text
            txt_enc_    = txt_enc2      if a.dualmod is not None and i in dualmod_nums else txt_enc
        if a.in_txt2 is not None:
            style_enc_  = style_enc2    if a.dualmod is not None and i in dualmod_nums else style_enc
        if a.in_img is not None and os.path.isfile(a.in_img):
            img_enc_    = img_enc2      if a.dualmod is not None and i in dualmod_nums else img_enc
        if a.in_txt0 is not None:
            not_enc_    = not_enc2      if a.dualmod is not None and i in dualmod_nums else not_enc
        if a.notext > 0:
            txtpic_enc_ = txt_plot_enc2 if a.dualmod is not None and i in dualmod_nums else txt_plot_enc
        model_clip_     = model_clip2   if a.dualmod is not None and i in dualmod_nums else model_clip
        if a.aest != 0:
            aest_       = aest2         if a.dualmod is not None and i in dualmod_nums else aest

        out_enc = model_clip_.encode_image(img_sliced)
        if a.aest != 0 and aest_ is not None:
            loss -= 0.001 * a.aest * aest_(out_enc).mean()
        if a.in_txt is not None: # input text
            loss +=  sign * sim_func(txt_enc_, out_enc, a.sim)
            if a.notext > 0:
                loss -= sign * a.notext * sim_func(txtpic_enc_, out_enc, a.sim)
        if a.in_txt2 is not None: # input text - style
            loss +=  sign * a.weight2 * sim_func(style_enc_, out_enc, a.sim)
        if a.in_txt0 is not None: # subtract text
            loss += -sign * 0.3 * sim_func(not_enc_, out_enc, a.sim)
        if a.in_img is not None and os.path.isfile(a.in_img): # input image
            loss +=  sign * 0.5 * sim_func(img_enc_, out_enc, a.sim)
        if a.sync > 0 and a.in_img is not None and os.path.isfile(a.in_img): # image composition
            prog_sync = (a.steps // a.opt_step - i) / (a.steps // a.opt_step)
            loss += prog_sync * a.sync * sim_loss(F.interpolate(img_out, sim_size, mode='bicubic', align_corners=True).float(), img_in, normalize=True).squeeze()
        if a.sharp != 0 and a.dwt is not True: # scharr|sobel|default
            loss -= a.sharp * derivat(img_out, mode='naiv')
            # loss -= a.sharp * derivat(img_sliced, mode='scharr')
        if a.enforce != 0:
            img_sliced = slice_imgs([image_f(noise)], a.samples, a.modsize, trform_f, a.align, a.macro)[0]
            out_enc2 = model_clip_.encode_image(img_sliced)
            loss -= a.enforce * sim_func(out_enc, out_enc2, a.sim)
            del out_enc2; torch.cuda.empty_cache()
        if a.expand > 0:
            global prev_enc
            if i > 0:
                loss += a.expand * sim_func(out_enc, prev_enc, a.sim)
            prev_enc = out_enc.detach() # .clone()

        del img_out, img_sliced, out_enc; torch.cuda.empty_cache()
        assert not isinstance(loss, int), ' Loss not defined, check the inputs'

        if a.prog is True:
            lr_cur = lr0 + (i / a.steps) * (lr1 - lr0)
            for g in optimizer.param_groups: 
                g['lr'] = lr_cur
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % a.opt_step == 0:
            with torch.no_grad():
                img = image_f(contrast=a.contrast).cpu().numpy()[0]
            # empirical tone mapping
            if (a.sync > 0 and a.in_img is not None):
                img = img **1.3
            elif a.sharp != 0:
                img = img ** (1 + a.sharp/2.)
            checkout(img, os.path.join(tempdir, '%04d.jpg' % (i // a.opt_step)), verbose=a.verbose)
            pbar.upd()

    pbar = ProgressBar(a.steps // a.opt_step)
    for i in range(a.steps):
        train(i)

    os.system('ffmpeg -v warning -y -i %s/\%%04d.jpg "%s.mp4"' % (tempdir, os.path.join(a.out_dir, out_name)))
    shutil.copy(img_list(tempdir)[-1], os.path.join(a.out_dir, '%s-%d.jpg' % (out_name, a.steps)))
    if a.save_pt is True:
        torch.save(params, '%s.pt' % os.path.join(a.out_dir, out_name))

if __name__ == '__main__':
    main()
