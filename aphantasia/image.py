import torch
import numpy as np
from aphantasia.utils import old_torch

import pywt
from pytorch_wavelets import DWTForward, DWTInverse
# from pytorch_wavelets import DTCWTForward, DTCWTInverse

def to_valid_rgb(image_f, colors=1., decorrelate=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]).astype("float32")
    color_correlation_svd_sqrt /= np.asarray([colors, 1., 1.]) # saturate, empirical
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    colcorr_t = torch.tensor(color_correlation_normalized.T).to(device)

    def _linear_decorrelate_color(tensor):
        t_permute = tensor.permute(0,2,3,1)
        t_permute = torch.matmul(t_permute, colcorr_t)
        tensor = t_permute.permute(0,3,1,2)
        return tensor

    def inner(*args, **kwargs):
        image = image_f(*args, **kwargs)
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)
    return inner
    
### DWT [wavelets]
   
def init_dwt(resume=None, shape=None, wave=None, colors=None):
    size = None
    wp_fake = pywt.WaveletPacket2D(data=np.zeros(shape[2:]), wavelet='db1', mode='symmetric')
    xfm = DWTForward(J=wp_fake.maxlevel, wave=wave, mode='symmetric').cuda()
    # xfm = DTCWTForward(J=lvl, biort='near_sym_b', qshift='qshift_b').cuda() # 4x more params, biort ['antonini','legall','near_sym_a','near_sym_b']
    ifm = DWTInverse(wave=wave, mode='symmetric').cuda() # symmetric zero periodization
    # ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda() # 4x more params, biort ['antonini','legall','near_sym_a','near_sym_b']
    if resume is None: # random init
        Yl_in, Yh_in = xfm(torch.zeros(shape).cuda())
        Ys = [torch.randn(*Y.shape).cuda() for Y in [Yl_in, *Yh_in]]
    elif isinstance(resume, str):
        if os.path.isfile(resume):
            if os.path.splitext(resume)[1].lower()[1:] in ['jpg','png','tif','bmp']:
                img_in = imread(resume)
                Ys = img2dwt(img_in, wave=wave, colors=colors)
                print(' loaded image', resume, img_in.shape, 'level', len(Ys)-1)
                size = img_in.shape[:2]
                wp_fake = pywt.WaveletPacket2D(data=np.zeros(size), wavelet='db1', mode='symmetric')
                xfm = DWTForward(J=wp_fake.maxlevel, wave=wave, mode='symmetric').cuda()
            else:
                Ys = torch.load(resume)
                Ys = [y.detach().cuda() for y in Ys]
        else: print(' Snapshot not found:', resume); exit()
    else:
        Ys = [y.cuda() for y in resume]
    # print('level', len(Ys)-1, 'low freq', Ys[0].cpu().numpy().shape)
    return Ys, xfm, ifm, size

def dwt_image(shape, wave='coif2', sharp=0.3, colors=1., resume=None):
    Ys, _, ifm, size = init_dwt(resume, shape, wave, colors)
    Ys = [y.requires_grad_(True) for y in Ys]
    scale = dwt_scale(Ys, sharp)

    def inner(shift=None, contrast=1.):
        image = ifm((Ys[0], [Ys[i+1] * float(scale[i]) for i in range(len(Ys)-1)]))
        image = image * contrast / image.std() # keep contrast, empirical *1.33
        return image

    return Ys, inner, size

def dwt_scale(Ys, sharp):
    scale = []
    [h0,w0] = Ys[1].shape[3:5]
    for i in range(len(Ys)-1):
        [h,w] = Ys[i+1].shape[3:5]
        scale.append( ((h0*w0)/(h*w)) ** (1.-sharp) )
        # print(i+1, Ys[i+1].shape)
    return scale

def img2dwt(img_in, wave='coif2', sharp=0.3, colors=1.):
    if not isinstance(img_in, torch.Tensor):
        img_in = torch.Tensor(img_in).cuda().permute(2,0,1).unsqueeze(0).float() / 255.
    img_in = un_rgb(img_in, colors=colors)
    with torch.no_grad():
        wp_fake = pywt.WaveletPacket2D(data=np.zeros(img_in.shape[2:]), wavelet='db1', mode='zero')
        lvl = wp_fake.maxlevel
        # print(img_in.shape, lvl)
        xfm = DWTForward(J=lvl, wave=wave, mode='symmetric').cuda()
        Yl_in, Yh_in = xfm(img_in.cuda())
        Ys = [Yl_in, *Yh_in]
    scale = dwt_scale(Ys, sharp)
    for i in range(len(Ys)-1):
        Ys[i+1] /= scale[i]
    return Ys

### FFT/RGB from Lucent library ###  https://github.com/greentfrapp/lucent

def pixel_image(shape, resume=None, sd=1., *noargs, **nokwargs):
    size = None
    if resume is None:
        tensor = torch.randn(*shape) * sd
    elif isinstance(resume, str):
        if os.path.isfile(resume):
            img_in = img_read(resume) / 255.
            tensor = torch.Tensor(img_in).permute(2,0,1).unsqueeze(0).float().cuda()
            tensor = un_rgb(tensor-0.5, colors=2.) # experimental
            size = img_in.shape[:2]
            print(resume, size)
        else: print(' Image not found:', resume); exit()
    else:
        if isinstance(resume, list): resume = resume[0]
        tensor = resume
    tensor = tensor.cuda().requires_grad_(True)

    def inner(shift=None, contrast=1.): # *noargs, **nokwargs
        image = tensor * contrast / tensor.std()
        return image
    return [tensor], inner, size # lambda: tensor

# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional frequency and later cut off 1 pixel
    w2 = (w+1)//2 if w%2 == 1 else w//2+1
    fx = np.fft.fftfreq(w)[:w2]
    return np.sqrt(fx * fx + fy * fy)

def resume_fft(resume=None, shape=None, decay=None, colors=1.6, sd=0.01):
    size = None
    if resume is None: # random init
        params_shape = [*shape[:3], shape[3]//2+1, 2] # [1,3,512,257,2] for 512x512 (2 for imaginary and real components)
        params = 0.01 * torch.randn(*params_shape).cuda()
    elif isinstance(resume, str):
        if os.path.isfile(resume):
            if os.path.splitext(resume)[1].lower()[1:] in ['jpg','png','tif','bmp']:
                img_in = img_read(resume)
                params = img2fft(img_in, decay, colors)
                size = img_in.shape[:2]
            else:
                params = torch.load(resume)
                if isinstance(params, list): params = params[0]
                params = params.detach().cuda()
            params *= sd
        else: print(' Snapshot not found:', resume); exit()
    else:
        if isinstance(resume, list): resume = resume[0]
        params = resume.cuda()
    return params, size

def fft_image(shape, sd=0.01, decay_power=1.0, resume=None): # decay ~ blur

    params, size = resume_fft(resume, shape, decay_power, sd=sd)
    spectrum_real_imag_t = params.requires_grad_(True)
    if size is not None: shape[2:] = size
    [h,w] = list(shape[2:])

    freqs = rfft2d_freqs(h,w)
    scale = 1. / np.maximum(freqs, 4./max(h,w)) ** decay_power
    scale *= np.sqrt(h*w)
    scale = torch.tensor(scale).float()[None, None, ..., None].cuda()

    def inner(shift=None, contrast=1.):
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if shift is not None:
            scaled_spectrum_t += scale * shift
        if old_torch():
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        else:
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        image = image * contrast / image.std() # keep contrast, empirical
        return image

    return [spectrum_real_imag_t], inner, size

def inv_sigmoid(x):
    eps = 1.e-12
    x = torch.clamp(x.double(), eps, 1-eps)
    y = torch.log(x/(1-x))
    return y.float()

def un_rgb(image, colors=1.):
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]).astype("float32")
    color_correlation_svd_sqrt /= np.asarray([colors, 1., 1.])
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    color_uncorrelate = np.linalg.inv(color_correlation_normalized)

    image = inv_sigmoid(image)
    t_permute = image.permute(0,2,3,1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_uncorrelate.T).cuda())
    image = t_permute.permute(0,3,1,2)
    return image

def un_spectrum(spectrum, decay_power):
    h = spectrum.shape[2]
    w = (spectrum.shape[3]-1)*2
    freqs = rfft2d_freqs(h, w)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w*h)
    scale = torch.tensor(scale).float()[None, None, ..., None].cuda()
    return spectrum / scale

def img2fft(img_in, decay=1., colors=1.):
    if isinstance(img_in, torch.Tensor):
        h, w = img_in.shape[2], img_in.shape[3]
    else:
        h, w = img_in.shape[0], img_in.shape[1]
        img_in = torch.Tensor(img_in).cuda().permute(2,0,1).unsqueeze(0) / 255.
    img_in = un_rgb(img_in, colors=colors)

    with torch.no_grad():
        if old_torch():
            spectrum = torch.rfft(img_in, 2, normalized=True) # 1.7
        else:
            spectrum = torch.fft.rfftn(img_in, s=(h, w), dim=[2,3], norm='ortho') # 1.8
            spectrum = torch.view_as_real(spectrum)
        spectrum = un_spectrum(spectrum, decay_power=decay)
        spectrum *= 500000. # [sic!!!]
    return spectrum
