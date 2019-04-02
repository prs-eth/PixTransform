# PixTransform

PyTorch implementation of the algorithm presented in [1]. The algorithm can be used to perform guided super-resolution, for instance:

<img align="center" width="400px" src="imgs/Frontpage.png">

The function `PixTransform` takes as input two images, the source image of size M x M
and a guide image of size C x N x N, and returns an upsampled version of the source image with size
N x N. The upsampling factor D, equal to N/M, must be an integer.
    
    predicted_target = PixTransform(source,guide)
    
additional variables can be passed to change the default parameters. For further details about the algorithm see [1]


### Getting started

##### Installation
clone this git repository and make sure that the following packages are installed:
* numpy
* matplotlib
* scipy
* Pytorch
* ProxTV (optional)
* tqdm

##### Example script
To run the algorithm on some sample images check the Jupyter Notebook file `process_examples.ipynb`.



###### References

[1] R. de Lutio, S. D'Aronco, J. D. Wegner, K. Schindler, "Guided Super-Resolution as a Learned Pixel-to-Pixel
Transformation", *arXiv*, 2019.
