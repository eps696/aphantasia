import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import shutil
import math
from collections import OrderedDict

try:
    from googletrans import Translator
    googletrans_ok = True
except ImportError as e:
    googletrans_ok = False

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import clip
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from aphantasia.utils import slice_imgs, derivat, aesthetic_model, txt_clean, checkout, old_torch
from aphantasia import transforms
from shader_expo import cppn_to_shader

from eps.progress_bar import ProgressBar
from eps.data_load import basename, img_list, img_read, file_list, save_cfg

clip_models = ['ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--in_img',  default=None, help='input image')
    parser.add_argument('-t',  '--in_txt',  default=None, help='input text')
    parser.add_argument('-t0', '--in_txt0', default=None, help='input text to subtract')
    parser.add_argument(       '--out_dir', default='_out')
    parser.add_argument('-r',  '--resume',  default=None, help='Input CPPN model (NPY file) to resume from')
    parser.add_argument('-s',  '--size',    default='512-512', help='Output resolution')
    parser.add_argument(       '--fstep',   default=1, type=int, help='Saving step')
    parser.add_argument('-tr', '--translate', action='store_true')
    parser.add_argument('-v',  '--verbose', action='store_true')
    parser.add_argument('-ex', '--export',  action='store_true', help="Only export shaders from resumed snapshot")
    # networks
    parser.add_argument('-l',  '--layers',  default=10, type=int, help='CPPN layers')
    parser.add_argument('-nf', '--nf',      default=24, type=int, help='num features') # 256
    parser.add_argument('-act', '--actfn',  default='unbias', choices=['unbias', 'comp', 'relu'], help='activation function')
    parser.add_argument('-dec', '--decim',  default=3, type=int, help='Decimal precision for export')
    # training
    parser.add_argument('-m',  '--model',   default='ViT-B/32', choices=clip_models, help='Select CLIP model to use')
    parser.add_argument('-dm', '--dualmod', default=None, type=int, help='Every this step use another CLIP ViT model')
    parser.add_argument(       '--steps',   default=200, type=int, help='Total iterations')
    parser.add_argument(       '--samples', default=50, type=int, help='Samples to evaluate')
    parser.add_argument('-lr', '--lrate',   default=0.003, type=float, help='Learning rate')
    parser.add_argument('-a',  '--align',   default='overscan', choices=['central', 'uniform', 'overscan'], help='Sampling distribution')
    parser.add_argument('-sh', '--sharp',   default=0, type=float)
    parser.add_argument('-tf', '--transform', action='store_true', help='use augmenting transforms?')
    parser.add_argument('-mc', '--macro',   default=0.4, type=float, help='Endorse macro forms 0..1; -1 = normal big')
    parser.add_argument(       '--aest',    default=0., type=float)
    a = parser.parse_args()
    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    if a.translate is True and googletrans_ok is not True: 
        print('\n Install googletrans module to enable translation!'); exit()
    if a.dualmod is not None: 
        a.model = 'ViT-B/32'
    return a


class ConvLayer(nn.Module):
    def __init__(self, nf_in, nf_out, act_fn='relu'):
        super().__init__()
        self.nf_in = nf_in
        self.conv = nn.Conv2d(nf_in, nf_out, 1, 1)
        if act_fn == 'comp':
            self.act_fn = self.composite_activation
        elif act_fn == 'unbias':
            self.act_fn = self.composite_activation_unbiased
        elif act_fn == 'relu':
            self.act_fn = self.relu_normalized
        else: # last layer (output)
            self.act_fn = torch.sigmoid
        with torch.no_grad(): # init
            self.conv.weight.normal_(0., math.sqrt(1./self.nf_in))
            self.conv.bias.uniform_(-.5, .5)
    
    def composite_activation(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)
    def composite_activation_unbiased(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x-0.45)/0.396], 1)
    def relu_normalized(self, x):
        x = F.relu(x)
        return (x-0.40)/0.58
    # https://colab.research.google.com/drive/1F1c2ouulmqys-GJBVBHn04I1UVWeexiB

    def forward(self, input):
        return self.act_fn(self.conv(input))
    
class CPPN(nn.Module):
    def __init__(self, nf_in=2, nf_hid=16, num_layers=9, nf_out=3, act_fn='unbias'): # unbias relu
        super().__init__()
        nf_hid_in = nf_hid if act_fn == 'relu' else nf_hid*2
        self.net = []
        self.net.append(ConvLayer(nf_in, nf_hid, act_fn))
        for i in range(num_layers-1):
            self.net.append(ConvLayer(nf_hid_in, nf_hid, act_fn))
        self.net.append(ConvLayer(nf_hid_in, nf_out, 'sigmoid'))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # [1,3,h,w]
        output = self.net(coords.cuda())
        return output

def load_cppn(file, verbose=True): # actfn='unbias'
    params = np.load(file, allow_pickle=True)
    nf = params[0].shape[-1]
    num_layers = len(params) // 2 - 1
    act_fn = 'relu' if params[0].shape[-1] == params[2].shape[-2] else 'unbias'
    snet = CPPN(2, nf, num_layers, 3, act_fn=act_fn).cuda()
    if verbose is True: print(' loaded:', file)
    if verbose is True: print(' .. %d vars, %d layers, %d nf, act %s' % (len(params), num_layers, nf, act_fn))
    keys = list(snet.state_dict().keys())
    assert len(keys) == len(params)
    cppn_dict = OrderedDict({})
    for lnum in range(0, len(keys), 2):
        cppn_dict[keys[lnum]] = np.transpose(torch.from_numpy(params[lnum]), (3,2,1,0))
        cppn_dict[keys[lnum+1]] = torch.from_numpy(params[lnum+1])
    snet.load_state_dict(cppn_dict)
    return snet

def get_mgrid(sideX, sideY):
    tensors = [np.linspace(-1, 1, num=sideY), np.linspace(-1, 1, num=sideX)]
    mgrid = np.stack(np.meshgrid(*tensors), axis=-1)
    mgrid = np.transpose(mgrid, (2,0,1))[np.newaxis]
    return mgrid

def export_gfx(model, out_name, mode, precision, size):
    shader = cppn_to_shader(model, mode=mode, verbose=False, fix_aspect=True, size=size, precision=precision)
    if mode == 'vvvv':     out_path = out_name + '.tfx'
    elif mode == 'buffer': out_path = out_name + '.txt'
    else:                  out_path = out_name + '-%s.glsl' % mode
    with open(out_path, 'wt') as f:
        f.write(shader)
    return out_path

def export_data(cppn_dict, out_name, size, decim=3, actfn='unbias', shaders=False, npy=True):
    if npy is True:     arrays = []
    if shaders is True: params = []
    keys = list(cppn_dict.keys())

    for lnum in range(0, len(keys), 2):
        w = cppn_dict[keys[lnum]].permute((3,2,1,0)).cpu().numpy()
        b = cppn_dict[keys[lnum+1]].cpu().numpy()
        if shaders is True: params.append({'weights': w, 'bias': b, 'activation': actfn})
        if npy is True: arrays += [w,b]

    if npy is True:
        np.save(out_name + '.npy', np.array(arrays, object))
    if shaders is True:
        export_gfx(params, out_name, 'td', decim, size)
        export_gfx(params, out_name, 'vvvv', decim, size)
        export_gfx(params, out_name, 'buffer', decim, size)
        export_gfx(params, out_name, 'bookofshaders', decim, size)
        export_gfx(params, out_name, 'shadertoy', decim, size)
    

def main():
    a = get_args()
    bx = 1.

    mgrid = get_mgrid(*a.size)
    mgrid = torch.from_numpy(mgrid.astype(np.float32)).cuda()

    # Load models
    if a.resume is not None and os.path.isfile(a.resume):
        snet = load_cppn(a.resume)
    else:
        snet = CPPN(mgrid.shape[1], a.nf, a.layers, 3, act_fn=a.actfn).cuda()
        print(' .. %d vars, %d layers, %d nf, act %s' % (len(snet.state_dict().keys()), a.layers, a.nf, a.actfn))

    if a.export is True:
        print('exporting')
        export_data(snet.state_dict(), a.resume.replace('.npy', ''), a.size, a.decim, a.actfn, shaders=True, npy=False)
        img = snet(mgrid).detach().cpu().numpy()[0]
        checkout(img, a.resume.replace('.npy', '.jpg'), verbose=False)
        exit(0)

    model_clip, _ = clip.load(a.model, jit=old_torch())
    try:
        a.modsize = model_clip.visual.input_resolution 
    except:
        a.modsize = 288 if a.model == 'RN50x4' else 384 if a.model == 'RN50x16' else 448 if a.model == 'RN50x64' else 224
    xmem = {'ViT-B/16':0.25, 'ViT-L/14':0.11, 'RN50':0.5, 'RN50x4':0.16, 'RN50x16':0.06, 'RN50x64':0.04, 'RN101':0.33}
    if a.model in xmem.keys():
        a.samples = int(a.samples * xmem[a.model])

    if a.dualmod is not None:
        model_clip2, _ = clip.load('ViT-B/16', jit=old_torch())
        a.samples = int(a.samples * 0.69) # second is vit-16
        dualmod_nums = list(range(a.steps))[a.dualmod::a.dualmod]
        print(' dual model every %d step' % a.dualmod)

    if a.aest != 0 and a.model in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        aest = aesthetic_model(a.model).cuda()
        if a.dualmod is not None:
            aest2 = aesthetic_model('ViT-B/16').cuda()
    
    def enc_text(txt, model_clip=model_clip):
        if txt is None or len(txt)==0: return None
        emb = model_clip.encode_text(clip.tokenize(txt).cuda()[:,:77])
        return emb.detach().clone()

    optimizer = torch.optim.Adam(snet.parameters(), a.lrate) # orig .00001, better 0.0001

    if a.transform is True:
        trform_f = transforms.trfm_fast  
        a.samples = int(a.samples * 0.95)
    else:
        trform_f = transforms.normalize()

    out_name = []
    if a.in_txt is not None:
        print(' ref text: ', basename(a.in_txt))
        if a.translate:
            translator = Translator()
            a.in_txt = translator.translate(a.in_txt, dest='en').text
            print(' translated to:', a.in_txt) 
        txt_enc = enc_text(a.in_txt)
        if a.dualmod is not None:
            txt_enc2 = enc_text(a.in_txt, model_clip2)
        out_name.append(txt_clean(a.in_txt))

    if a.in_txt0 is not None:
        print(' no text: ', basename(a.in_txt0))
        if a.translate:
            translator = Translator()
            a.in_txt0 = translator.translate(a.in_txt0, dest='en').text
            print(' translated to:', a.in_txt0) 
        not_enc = enc_text(a.in_txt0)
        if a.dualmod is not None:
            not_enc2 = enc_text(a.in_txt0, model_clip2)

    img_enc = None
    if a.in_img is not None and os.path.isfile(a.in_img):
        print(' ref image:', basename(a.in_img))
        img_in = torch.from_numpy(img_read(a.in_img)/255.).unsqueeze(0).permute(0,3,1,2).cuda()
        in_sliced = slice_imgs([img_in], a.samples, a.modsize, transforms.normalize(), a.align)[0]
        img_enc = model_clip.encode_image(in_sliced).detach().clone()
        if a.dualmod is not None:
            img_enc2 = model_clip2.encode_image(in_sliced).detach().clone()
        del img_in, in_sliced; torch.cuda.empty_cache()
        out_name.append(basename(a.in_img).replace(' ', '_'))

    # Prepare dirs
    sfx = '-l%d-n%d' % (a.layers, a.nf)
    if a.dualmod is not None:    sfx += '-dm%d' % a.dualmod
    if a.aest != 0:              sfx += '-ae%.2g' % a.aest
    workdir = os.path.join(a.out_dir, 'cppn')
    out_name = os.path.join(workdir, '-'.join(out_name) + sfx)
    tempdir = out_name
    os.makedirs(out_name, exist_ok=True)
    print(a.samples)
    
    def train(i, img_enc=None):
        loss = 0
        img_out = snet(mgrid)

        txt_enc_      = txt_enc2      if a.dualmod is not None and i in dualmod_nums else txt_enc
        if a.in_img is not None and os.path.isfile(a.in_img):
            img_enc_  = img_enc2      if a.dualmod is not None and i in dualmod_nums else img_enc
        if a.in_txt0 is not None:
            not_enc_  = not_enc2      if a.dualmod is not None and i in dualmod_nums else not_enc
        model_clip_   = model_clip2   if a.dualmod is not None and i in dualmod_nums else model_clip
        if a.aest != 0:
            aest_ = aest2             if a.dualmod is not None and i in dualmod_nums else aest

        imgs_sliced = slice_imgs([img_out], a.samples, a.modsize, trform_f, a.align, a.macro)
        out_enc = model_clip_.encode_image(imgs_sliced[-1])
        if a.aest != 0 and aest_ is not None:
            loss -= 0.001 * a.aest * aest_(out_enc).mean()
        if a.in_txt is not None:
            loss -= torch.cosine_similarity(txt_enc_, out_enc, dim=-1).mean()
        if a.in_txt0 is not None:
            loss += 0.5 * torch.cosine_similarity(not_enc_, out_enc, dim=-1).mean()
        if a.in_img is not None and os.path.isfile(a.in_img):
            loss -= torch.cosine_similarity(img_enc_, out_enc, dim=-1).mean()
        if a.sharp != 0: # mode = scharr|sobel|default
            loss -= a.sharp * derivat(img_out, mode='sobel')
        del img_out, imgs_sliced, out_enc; torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % a.fstep == 0:
            with torch.no_grad():
                img = snet(mgrid).cpu().numpy()[0]
            fname = os.path.join(tempdir, '%04d' % (i // a.fstep))
            checkout(img, fname + '.jpg', verbose=a.verbose)
            export_data(snet.state_dict(), fname, a.size, a.decim)
        return 

    pbar = ProgressBar(a.steps)
    for i in range(a.steps):
        log = train(i, img_enc)
        pbar.upd(log)

    export_data(snet.state_dict(), out_name, a.size, a.decim, shaders=True)
    os.system('ffmpeg -v warning -y -i %s\%%04d.jpg -c:v mjpeg -pix_fmt yuvj444p -dst_range 1 -q:v 2 "%s.avi"' % (tempdir, out_name))
    shutil.copy(img_list(tempdir)[-1], out_name + '-%d.jpg' % a.steps)
    # shutil.rmtree(tempdir)


if __name__ == '__main__':
    main()
