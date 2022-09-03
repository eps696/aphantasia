import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import math
import numpy as np

import torch

from clip_fft import to_valid_rgb, fft_image
from aphantasia.utils import basename, file_list, checkout
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from aphantasia.progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from aphantasia.progress_bar import ProgressBar

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', default='pt')
    parser.add_argument('-o', '--out_dir', default='_out')
    parser.add_argument('-l', '--length',  default=None, type=int, help='Total length in sec')
    parser.add_argument('-s', '--steps',   default=25, type=int, help='Override length')
    parser.add_argument(       '--fps',     default=25, type=int)
    parser.add_argument(       '--contrast', default=1.1, type=float)
    parser.add_argument(       '--colors',  default=1.8, type=float)
    parser.add_argument('-d',  '--decay',   default=1.5, type=float)
    parser.add_argument('-v', '--verbose', default=True, type=bool)
    a = parser.parse_args()
    return a

def read_pt(file):
    return torch.load(file)[0].cuda()

def main():
    a = get_args()
    tempdir = os.path.join(a.out_dir, 'a')
    os.makedirs(tempdir, exist_ok=True)
    
    ptfiles = file_list(a.in_dir, 'pt')

    ptest = torch.load(ptfiles[0])
    if isinstance(ptest, list): ptest = ptest[0]
    shape = [*ptest.shape[:3], (ptest.shape[3]-1)*2]

    vsteps = a.lsteps if a.length is None else int(a.length * a.fps / count)
    pbar = ProgressBar(vsteps * len(ptfiles))
    for px in range(len(ptfiles)):
        params1 = read_pt(ptfiles[px])
        params2 = read_pt(ptfiles[(px+1) % len(ptfiles)])

        params, image_f, _ = fft_image(shape, resume=params1, sd=1., decay_power=a.decay)
        image_f = to_valid_rgb(image_f, colors = a.colors)

        for i in range(vsteps):
            with torch.no_grad():
                x = i/vsteps # math.sin(1.5708 * i/vsteps)
                img = image_f((params2 - params1) * x, contrast=a.contrast).cpu().numpy()[0]
            checkout(img, os.path.join(tempdir, '%05d.jpg' % (px * vsteps + i)), verbose=a.verbose)
            pbar.upd()

    os.system('ffmpeg -v warning -y -i %s/\%%05d.jpg "%s-pts.mp4"' % (tempdir, a.in_dir))


if __name__ == '__main__':
    main()
