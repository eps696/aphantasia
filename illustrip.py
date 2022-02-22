# coding: UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import shutil
import PIL
import time
from imageio import imread, imsave

try:
    from googletrans import Translator
    googletrans_ok = True
except ImportError as e:
    googletrans_ok = False

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as T

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from aphantasia.image import to_valid_rgb, fft_image, resume_fft, pixel_image
from aphantasia.utils import slice_imgs, derivat, sim_func, slerp, basename, file_list, img_list, img_read, pad_up_to, txt_clean, latent_anima, cvshow, checkout, save_cfg, old_torch
from aphantasia import transforms
from depth import depth
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from aphantasia.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from aphantasia.progress_bar import ProgressBar

clip_models = ['ViT-B/16', 'ViT-B/32', 'RN50', 'RN50x4', 'RN50x16', 'RN101']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',  '--size',    default='1280-720', help='Output resolution')
    parser.add_argument('-t',  '--in_txt',  default=None, help='Text string or file to process (main topic)')
    parser.add_argument('-pre', '--in_txt_pre', default=None, help='Prefix for input text')
    parser.add_argument('-post', '--in_txt_post', default=None, help='Postfix for input text')
    parser.add_argument('-t2', '--in_txt2', default=None, help='Text string or file to process (style)')
    parser.add_argument('-t0', '--in_txt0', default=None, help='input text to subtract')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images')
    parser.add_argument('-w0', '--weight0', default=0.3, type=float, help='weight for subtraction')
    parser.add_argument('-w2', '--weight2', default=0.5, type=float, help='weight for style')
    parser.add_argument('-wi', '--weight_img', default=0.5, type=float, help='weight for images')
    parser.add_argument('-r',  '--resume',  default=None, help='Resume from saved params or from an image')
    parser.add_argument(       '--out_dir', default='_out')
    parser.add_argument('-tr', '--translate', action='store_true', help='Translate with Google Translate')
    parser.add_argument(       '--invert',  action='store_true', help='Invert criteria')
    parser.add_argument('-v',  '--verbose',    dest='verbose', action='store_true')
    parser.add_argument('-nv', '--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    # training
    parser.add_argument(       '--gen',     default='RGB', help='Generation (optimization) method: FFT or RGB')
    parser.add_argument('-m',  '--model',   default='ViT-B/32', choices=clip_models, help='Select CLIP model to use')
    parser.add_argument(       '--steps',   default=300, type=int, help='Iterations (frames) per scene (text line)')
    parser.add_argument(       '--samples', default=100, type=int, help='Samples to evaluate per frame')
    parser.add_argument('-lr', '--lrate',   default=1, type=float, help='Learning rate')
    # motion
    parser.add_argument('-ops', '--opt_step', default=1, type=int, help='How many optimizing steps per save/transform step')
    parser.add_argument('-sm', '--smooth',  action='store_true', help='Smoothen interframe jittering for FFT method')
    parser.add_argument('-it', '--interpol', default=True, help='Interpolate topics? (or change by cut)')
    parser.add_argument(       '--fstep',   default=100, type=int, help='How many frames before changing motion')
    parser.add_argument(       '--scale',   default=0.012, type=float)
    parser.add_argument(       '--shift',   default=10., type=float, help='in pixels')
    parser.add_argument(       '--angle',   default=0.8, type=float, help='in degrees')
    parser.add_argument(       '--shear',   default=0.4, type=float)
    parser.add_argument(       '--anima',   default=True, help='Animate motion')
    # depth
    parser.add_argument('-d',  '--depth',   default=0, type=float, help='Add depth with such strength, if > 0')
    parser.add_argument(       '--tridepth', action='store_true', help='process depth 3 times [mirrored]')
    parser.add_argument(   '--depth_model', default='AdaBins_nyu.pt', help='AdaBins model path')
    parser.add_argument(   '--depth_mask',  default='depth/mask.jpg', help='depth mask path')
    parser.add_argument(   '--depth_dir',   default=None, help='Directory to save depth, if not None')
    # tweaks
    parser.add_argument('-a',  '--align',   default='overscan', choices=['central', 'uniform', 'overscan', 'overmax'], help='Sampling distribution')
    parser.add_argument('-tf', '--transform', default='fast', choices=['none', 'fast', 'custom', 'elastic'], help='augmenting transforms')
    parser.add_argument('-opt', '--optimizer', default='adam', choices=['adam', 'adam_custom', 'adamw', 'adamw_custom'], help='Optimizer')
    parser.add_argument(       '--fixcontrast', action='store_true', help='Required for proper resuming from image')
    parser.add_argument(       '--contrast', default=1.2, type=float)
    parser.add_argument(       '--colors',  default=2.3, type=float)
    parser.add_argument('-sh', '--sharp',   default=0, type=float)
    parser.add_argument('-mc', '--macro',   default=0.3, type=float, help='Endorse macro forms 0..1 ')
    parser.add_argument('-e',  '--enforce', default=0, type=float, help='Enforce details (by boosting similarity between two parallel samples)')
    parser.add_argument('-x',  '--expand',  default=0, type=float, help='Boosts diversity (by enforcing difference between prev/next samples)')
    parser.add_argument('-n',  '--noise',   default=2., type=float, help='Add noise to make composition sparse (FFT only)') # 0.04
    parser.add_argument(       '--sim',     default='mix', help='Similarity function (angular/spherical/mixed; None = cossim)')
    parser.add_argument(       '--rem',     default=None, help='Dummy text to add to project name')
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    a.gen = a.gen.upper()
    a.invert = -1. if a.invert is True else 1.
    
    # Overriding some parameters, depending on other settings
    if a.gen == 'RGB':
        a.smooth = False
        a.align = 'overscan'
        if a.resume is not None: a.fixcontrast = True
    if a.model == 'ViT-B/16': a.sim = 'cossim'

    if a.translate is True and googletrans_ok is not True: 
        print('\n Install googletrans module to enable translation!'); exit()
    
    return a

def depth_transform(img_t, depth_infer, depth_mask, size, depthX=0, scale=1., shift=[0,0], colors=1, depth_dir=None, save_num=0):
    size2 = [s//2 for s in size]
    if not isinstance(scale, float): scale = float(scale[0])
    # d X/Y define the origin point of the depth warp, effectively a "3D pan zoom", [-1..1]
    # plus = look ahead, minus = look aside
    dX = 100. * shift[0] / size[1]
    dY = 100. * shift[1] / size[0]
    # dZ = movement direction: 1 away (zoom out), 0 towards (zoom in), 0.5 stay
    dZ = 0.5 + 32. * (scale-1)
    img = depth.depthwarp(img_t, depth_infer, depth_mask, size2, depthX, [dX,dY], dZ, save_path=depth_dir, save_num=save_num)
    return img

def frame_transform(img, size, angle, shift, scale, shear):
    if old_torch(): # 1.7.1
        img = T.functional.affine(img, angle, tuple(shift), scale, shear, fillcolor=0, resample=PIL.Image.BILINEAR)
        img = T.functional.center_crop(img, size)
        img = pad_up_to(img, size)
    else: # 1.8+
        img = T.functional.affine(img, angle, tuple(shift), scale, shear, fill=0, interpolation=T.InterpolationMode.BILINEAR)
        img = T.functional.center_crop(img, size) # on 1.8+ also pads
    return img

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

    if a.translate:
        translator = Translator()

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

    def enc_text(txt):
        if a.translate:
            txt = translator.translate(txt, dest='en').text
        emb = model_clip.encode_text(clip.tokenize(txt).cuda()[:77])
        return emb.detach().clone()

    def enc_image(img_file):
        img_t = torch.from_numpy(img_read(img_file)/255.).unsqueeze(0).permute(0,3,1,2).cuda()[:,:3,:,:]
        in_sliced = slice_imgs([img_t], a.samples, a.modsize, transforms.normalize(), a.align)[0]
        emb = model_clip.encode_image(in_sliced)
        return emb.detach().clone()

    # Encode inputs
    count = 0
    texts = []
    styles = []
    images = []
    
    if a.in_txt is not None:
        if os.path.isfile(a.in_txt):
            with open(a.in_txt, 'r', encoding="utf-8") as f:
                texts = f.readlines()
                texts = [tt.strip() for tt in texts if len(tt.strip()) > 0 and tt[0] != '#']
        else:
            texts = [a.in_txt]
    if a.in_txt_pre is not None:
        texts = [' '.join([a.in_txt_pre, tt]).strip() for tt in texts]
    if a.in_txt_post is not None:
        texts = [' '.join([tt, a.in_txt_post]).strip() for tt in texts]
    key_txt_encs = [enc_text(txt) for txt in texts]
    count = max(count, len(key_txt_encs))

    if a.in_txt2 is not None:
        if os.path.isfile(a.in_txt2):
            with open(a.in_txt2, 'r', encoding="utf-8") as f:
                styles = f.readlines()
                styles = [tt.strip() for tt in styles if len(tt.strip()) > 0 and tt[0] != '#']
        else:
            styles = [a.in_txt2]
    key_styl_encs = [enc_text(style) for style in styles]
    count = max(count, len(key_styl_encs))

    if a.in_img is not None and os.path.exists(a.in_img):
        images = file_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
    key_img_encs = [enc_image(image) for image in images]
    count = max(count, len(key_img_encs))
    
    assert count > 0, "No inputs found!"
    
    if a.in_txt0 is not None:
        if a.verbose is True: print(' subtract text:', a.in_txt0)
        if a.translate:
            a.in_txt0 = translator.translate(a.in_txt0, dest='en').text
        anti_txt_encs = [enc_text(txt) for txt in a.in_txt0.split('.')]

    if a.verbose is True: print(' samples:', a.samples)

    global params_tmp
    shape = [1, 3, *a.size]

    if a.gen == 'RGB':
        params_tmp, _, sz = pixel_image(shape, a.resume)
        params_tmp = params_tmp[0].cuda().detach()
    else:
        params_tmp, sz = resume_fft(a.resume, shape, decay=1.5, sd=1)
    if sz is not None: a.size = sz

    if a.depth != 0:
        depth_infer, depth_mask = depth.init_adabins(size=a.size, model_path=a.depth_model, mask_path=a.depth_mask, tridepth=a.tridepth)
        if a.depth_dir is not None:
            os.makedirs(a.depth_dir, exist_ok=True)
            print(' depth dir:', a.depth_dir)

    steps = a.steps
    glob_steps = count * steps
    if glob_steps == a.fstep: a.fstep = glob_steps // 2 # otherwise no motion

    workname = basename(a.in_txt) if a.in_txt is not None else basename(a.in_img)
    workname = txt_clean(workname)
    workdir = os.path.join(a.out_dir, workname + '-%s' % a.gen.lower())
    if a.rem is not None:        workdir += '-%s' % a.rem
    if 'RN' in a.model.upper():  workdir += '-%s' % a.model
    tempdir = os.path.join(workdir, 'ttt')
    os.makedirs(tempdir, exist_ok=True)
    save_cfg(a, workdir)
    if a.in_txt is not None and os.path.isfile(a.in_txt):
        shutil.copy(a.in_txt, os.path.join(workdir, os.path.basename(a.in_txt)))
    if a.in_txt2 is not None and os.path.isfile(a.in_txt2):
        shutil.copy(a.in_txt2, os.path.join(workdir, os.path.basename(a.in_txt2)))

    midp = 0.5
    if a.anima:
        if a.gen == 'RGB': # zoom in
            m_scale = latent_anima([1], glob_steps, a.fstep, uniform=True, cubic=True, start_lat=[-0.3], verbose=False)
            m_scale = 1 + (m_scale + 0.3) * a.scale
        else:
            m_scale = latent_anima([1], glob_steps, a.fstep, uniform=True, cubic=True, start_lat=[0.6],  verbose=False)
            m_scale = 1 - (m_scale-0.6) * a.scale
        m_shift = latent_anima([2], glob_steps, a.fstep, uniform=True, cubic=True, start_lat=[midp,midp], verbose=False)
        m_angle = latent_anima([1], glob_steps, a.fstep, uniform=True, cubic=True, start_lat=[midp],    verbose=False)
        m_shear = latent_anima([1], glob_steps, a.fstep, uniform=True, cubic=True, start_lat=[midp],    verbose=False)
        m_shift = (midp-m_shift) * a.shift * abs(m_scale-1) / a.scale
        m_angle = (midp-m_angle) * a.angle * abs(m_scale-1) / a.scale
        m_shear = (midp-m_shear) * a.shear * abs(m_scale-1) / a.scale
    
    def get_encs(encs, num):
        cnt = len(encs)
        if cnt == 0: return []
        enc_1 = encs[min(num,   cnt-1)]
        enc_2 = encs[min(num+1, cnt-1)]
        return slerp(enc_1, enc_2, steps)

    prev_enc = 0
    def process(num):
        global params_tmp, opt_state, params, image_f, optimizer

        if a.interpol is True: # linear topics interpolation
            txt_encs  = get_encs(key_txt_encs,  num)
            styl_encs = get_encs(key_styl_encs, num)
            img_encs  = get_encs(key_img_encs,  num)
        else: # change by cut
            txt_encs  = [key_txt_encs[min(num,  len(key_txt_encs)-1)][0]]  * steps if len(key_txt_encs)  > 0 else []
            styl_encs = [key_styl_encs[min(num, len(key_styl_encs)-1)][0]] * steps if len(key_styl_encs) > 0 else []
            img_encs  = [key_img_encs[min(num,  len(key_img_encs)-1)][0]]  * steps if len(key_img_encs)  > 0 else []
        
        if a.verbose is True: 
            if len(texts)  > 0: print(' ref text: ',  texts[min(num, len(texts)-1)][:80])
            if len(styles) > 0: print(' ref style: ', styles[min(num, len(styles)-1)][:80])
            if len(images) > 0: print(' ref image: ', basename(images[min(num, len(images)-1)])[:80])
        
        pbar = ProgressBar(steps)
        for ii in range(steps):
            glob_step = num * steps + ii # save/transform
            
            txt_enc  = txt_encs[ii % len(txt_encs)].unsqueeze(0)   if len(txt_encs)  > 0 else None
            styl_enc = styl_encs[ii % len(styl_encs)].unsqueeze(0) if len(styl_encs) > 0 else None
            img_enc  = img_encs[ii % len(img_encs)].unsqueeze(0)   if len(img_encs)  > 0 else None

            # MOTION: transform frame, reload params

            scale = m_scale[glob_step]    if a.anima else 1 + a.scale
            shift = m_shift[glob_step]    if a.anima else [0, a.shift]
            angle = m_angle[glob_step][0] if a.anima else a.angle
            shear = m_shear[glob_step][0] if a.anima else a.shear

            if a.gen == 'RGB':
                if a.depth > 0:
                    params_tmp = depth_transform(params_tmp, depth_infer, depth_mask, a.size, a.depth, scale, shift, a.colors, a.depth_dir, glob_step)
                params_tmp = frame_transform(params_tmp, a.size, angle, shift, scale, shear)
                params, image_f, _ = pixel_image([1, 3, *a.size], resume=params_tmp)
                img_tmp = None

            else: # FFT
                if old_torch(): # 1.7.1
                    img_tmp = torch.irfft(params_tmp, 2, normalized=True, signal_sizes=a.size)
                    if a.depth > 0:
                        img_tmp = depth_transform(img_tmp, depth_infer, depth_mask, a.size, a.depth, scale, shift, a.colors, a.depth_dir, glob_step)
                    img_tmp = frame_transform(img_tmp, a.size, angle, shift, scale, shear)
                    params_tmp = torch.rfft(img_tmp, 2, normalized=True)
                else: # 1.8+
                    if type(params_tmp) is not torch.complex64:
                        params_tmp = torch.view_as_complex(params_tmp)
                    img_tmp = torch.fft.irfftn(params_tmp, s=a.size, norm='ortho')
                    if a.depth > 0:
                        img_tmp = depth_transform(img_tmp, depth_infer, depth_mask, a.size, a.depth, scale, shift, a.colors, a.depth_dir, glob_step)
                    img_tmp = frame_transform(img_tmp, a.size, angle, shift, scale, shear)
                    params_tmp = torch.fft.rfftn(img_tmp, s=a.size, dim=[2,3], norm='ortho')
                    params_tmp = torch.view_as_real(params_tmp)
                params, image_f, _ = fft_image([1, 3, *a.size], sd=1, resume=params_tmp)

            if a.optimizer.lower() == 'adamw':
                optimizer = torch.optim.AdamW(params, a.lrate, weight_decay=0.01)
            elif a.optimizer.lower() == 'adamw_custom':
                optimizer = torch.optim.AdamW(params, a.lrate, weight_decay=0.01, betas=(.0,.999), amsgrad=True)
            elif a.optimizer.lower() == 'adam':
                optimizer = torch.optim.Adam(params, a.lrate)
            else: # adam_custom
                optimizer = torch.optim.Adam(params, a.lrate, betas=(.0,.999))
            image_f = to_valid_rgb(image_f, colors = a.colors)
            del img_tmp

            if a.smooth is True and num + ii > 0:
                optimizer.load_state_dict(opt_state)

            ### optimization
            for ss in range(a.opt_step):
                loss = 0

                noise = a.noise * (torch.rand(1, 1, a.size[0], a.size[1]//2+1, 1)-0.5).cuda() if a.noise>0 else 0.
                img_out = image_f(noise, fixcontrast=a.fixcontrast)
                
                img_sliced = slice_imgs([img_out], a.samples, a.modsize, trform_f, a.align, a.macro)[0]
                out_enc = model_clip.encode_image(img_sliced)

                if a.gen == 'RGB': # empirical hack
                    loss += abs(img_out.mean((2,3)) - 0.45).mean() # fix brightness
                    loss += abs(img_out.std((2,3)) - 0.17).mean() # fix contrast

                if txt_enc is not None:
                    loss -= a.invert * sim_func(txt_enc, out_enc, a.sim)
                if styl_enc is not None:
                    loss -= a.weight2 * sim_func(styl_enc, out_enc, a.sim)
                if img_enc is not None:
                    loss -= a.weight_img * sim_func(img_enc, out_enc, a.sim)
                if a.in_txt0 is not None: # subtract text
                    for anti_txt_enc in anti_txt_encs:
                        loss += 0.3 * sim_func(anti_txt_enc, out_enc, a.sim)
                if a.sharp != 0: # scharr|sobel|naive
                    loss -= a.sharp * derivat(img_out, mode='naive')
                if a.enforce != 0:
                    img_sliced = slice_imgs([image_f(noise, fixcontrast=a.fixcontrast)], a.samples, a.modsize, trform_f, a.align, a.macro)[0]
                    out_enc2 = model_clip.encode_image(img_sliced)
                    loss -= a.enforce * sim_func(out_enc, out_enc2, a.sim)
                    del out_enc2; torch.cuda.empty_cache()
                if a.expand > 0:
                    global prev_enc
                    if ii > 0:
                        loss += a.expand * sim_func(prev_enc, out_enc, a.sim)
                    prev_enc = out_enc.detach().clone()
                del img_out, img_sliced, out_enc; torch.cuda.empty_cache()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ### save params & frame

            params_tmp = params[0].detach().clone()
            if a.smooth is True:
                opt_state = optimizer.state_dict()

            with torch.no_grad():
                img_t = image_f(contrast=a.contrast, fixcontrast=a.fixcontrast)[0].permute(1,2,0)
                img_np = torch.clip(img_t*255, 0, 255).cpu().numpy().astype(np.uint8)
            imsave(os.path.join(tempdir, '%06d.jpg' % glob_step), img_np, quality=95)
            if a.verbose is True: cvshow(img_np)
            del img_t, img_np
            pbar.upd()

        params_tmp = params[0].detach().clone()
        
    glob_start = time.time()
    try:
        for i in range(count):
            process(i)
    except KeyboardInterrupt:
        pass

    os.system('ffmpeg -v warning -y -i %s/\%%06d.jpg "%s.mp4"' % (tempdir, os.path.join(workdir, workname)))


if __name__ == '__main__':
    main()
