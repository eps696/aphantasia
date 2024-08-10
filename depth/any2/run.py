import os
import argparse
import cv2
import numpy as np
# import matplotlib

import torch

from deptha2.dpt import DepthAnythingV2

from eps import img_list, img_read, basename, progbar

parser = argparse.ArgumentParser(description='Depth Anything V2')
parser.add_argument('-i', '--input', default='_in', help='Input image or folder')
parser.add_argument('-o', '--out_dir', default='_out')
parser.add_argument('-md','--maindir', default='./', help='Main directory')
parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('-sz', '--size', type=int, default=768) # 518
parser.add_argument('--seed',          default=None, type=int, help='Random seed')
# parser.add_argument('--pre', action='store_true', help='display combined mix')
parser.add_argument('-v',  '--verbose', action='store_true')
a = parser.parse_args()

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
    
def main():
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    depth_anything = DepthAnythingV2(**model_configs[a.encoder])
    depth_anything.load_state_dict(torch.load(os.path.join(a.maindir, 'models', f'depth_anything_v2_{a.encoder}.pth'), map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    paths = [a.input] if os.path.isfile(a.input) else img_list(a.input)
    pbar = progbar(len(paths))
    for k, path in enumerate(paths):
        img_in = cv2.imread(path)

        depth = depth_anything.infer_image(img_in, a.size)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        # depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # if a.pre:
            # split_region = np.ones((img_in.shape[0], 50, 3), dtype=np.uint8) * 255
            # depth = cv2.hconcat([img_in, split_region, depth])
            
        cv2.imwrite(os.path.join(a.out_dir, basename(path) + '.png'), depth)
        pbar.upd()


if __name__ == '__main__':
    main()
