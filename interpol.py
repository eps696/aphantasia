import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import math
import numpy as np
from imageio import imsave

import torch

from clip_fft import to_valid_rgb, fft_image, cvshow
from utils import basename, file_list
try: # progress bar for notebooks 
    get_ipython().__class__.__name__
    from progress_bar import ProgressIPy as ProgressBar
except: # normal console
    from progress_bar import ProgressBar

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', default='pt')
    parser.add_argument('--out_dir', default='_out')
    parser.add_argument('-l',  '--length',  default=5, type=int, help='Total length in sec')
    parser.add_argument('-s', '--size', default='1280-720', help='Output resolution')
    parser.add_argument('-v', '--verbose', default=True, type=bool)
    a = parser.parse_args()

    if a.size is not None: a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size)==1: a.size = a.size * 2
    return a

def read_pt(file):
    return torch.load(file)[0].cuda()

def main():
    a = get_args()
    tempdir = os.path.join(a.out_dir, 'a')
    os.makedirs(tempdir, exist_ok=True)
    
    ptfiles = file_list(a.in_dir, 'pt')

    vsteps = int(a.length * 25 / len(ptfiles)) # 25 fps
    pbar = ProgressBar(vsteps * len(ptfiles))
    for px in range(len(ptfiles)):
        params1 = read_pt(ptfiles[px])
        params2 = read_pt(ptfiles[(px+1) % len(ptfiles)])

        params, image_f = fft_image([1, 3, *a.size], resume=params1)
        image_f = to_valid_rgb(image_f)

        for i in range(vsteps):
            with torch.no_grad():
                img = image_f((params2 - params1) * math.sin(1.5708 * i/vsteps)**2)[0].permute(1,2,0)
                img = torch.clip(img*255, 0, 255).cpu().numpy().astype(np.uint8)
            imsave(os.path.join(tempdir, '%05d.jpg' % (px * vsteps + i)), img)
            if a.verbose is True: cvshow(img)
            pbar.upd()

    os.system('ffmpeg -v warning -y -i %s\%%05d.jpg -c:v mjpeg -q:v 2 "%s-pts.avi"' % (tempdir, a.in_dir))


if __name__ == '__main__':
    main()
