# Aphantasia

<p align='center'><img src='_out/Aphantasia.jpg' /></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eps696/aphantasia/blob/master/Aphantasia.ipynb)

This is text-to-image tool, supplementary to the art piece of the same name.   
It's based on [CLIP] model, with FFT parameterizer from [Lucent] library as a generator. 

*[Aphantasia] is the inability to visualize mental images, the deprivation of visual dreams.  
The image in the header is how this tool sees it.*

## Features
* generating massive detailed textures, a la deepdream
* fast convergence!
* fullHD/4K resolutions and above
* complex queries:
	* image and/or text as main prompts
	* additional text prompts for fine details and to subtract (avoid) topics
	* criteria inversion (show "the opposite")
* saving/loading FFT params to resume processing
* can use both CLIP models at once (ViT and RN50)

## Operations

* Generate an image from the text prompt (set size as you wish):
```
python clip_fft.py -t "the text" --size 1280-720
```
* Reproduce an image:
```
python clip_fft.py -i theimage.jpg --sync 0.3
```
`--sync X` argument (X = from 0 to 1) enables [SSIM] loss to keep the composition and details of the original image. 

You can combine both text and image prompts.  
Use `--translate` option to process non-English languages. 

* Set more specific query:
```
python clip_fft.py -t "macro figures" -t2 "micro details" -t0 "avoid this" --size 1280-720 
```

`--steps N` sets iterations count. 50-100 is enough for a starter; 500-1000 would elaborate it more thoroughly.  
`--samples N` sets amount of the image cuts (samples), processed at one step. With more samples you can set fewer iterations for similar result (and vice versa). 200/200 is a good guess.  
`--fstep N` tells to save every Nth frame (useful with high iterations).  
`--dual` turns on optimisation with both CLIP models (a bit different results).  
`--invert` negates the whole criteria, if you fancy checking "totally opposite".

`--uniform` mode ('on' by default) sets random sampling, usually producing seamlessly tileable textures. Set it to `False`, if you need more centered composition.  
`--save_pt myfile.pt` will save FFT params, to resume for next query with `--resume myfile.pt`.

## Credits

[CLIP], [original paper] 
Copyright (c) 2021 OpenAI

Thanks to [Ryan Murdock] and [Jonathan Fly] for ideas.

[Aphantasia]: <https://en.wikipedia.org/wiki/Aphantasia>
[CLIP]: <https://openai.com/blog/clip>
[Lucent]: <https://github.com/greentfrapp/lucent>
[SSIM]: <https://github.com/Po-Hsun-Su/pytorch-ssim>
[Ryan Murdock]: <https://rynmurdock.github.io/>
[Jonathan Fly]: <https://twitter.com/jonathanfly>
[original paper]: <https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf>