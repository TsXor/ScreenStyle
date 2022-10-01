# ScreenVAE

Pytorch implementation for screentone encoding. 
For example, given the manga image with screentone, our model is able to generate feature representations which is plain within the region with same screentone and can also reconstruct the original manga image. 

**Note**: The current software works well with PyTorch 1.1+. 

## Example results


## Prerequisites
- Linux 
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone https://github.com/msxie92/ScreenStyle.git
cd ScreenStyle/screenVAE
```
- Install PyTorch and dependencies from http://pytorch.org

### Visulization
- Generate screentones by sampling in the intermediate domain:
```
python visual.py 
```

Examples:

![visual](examples/visual.png)

### Screentone Editing
- Edit screentones by modifying the value in the intermediate domain:
```
python gui_edit.py 
```

Examples:

![original](examples/manga.png)
![edited](examples/edited.png)

## Models
Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1OBxWHjijMwi9gfTOfDiFiHRZA_CXNSWr/view?usp=sharing) and place under checkpoints/ScreenVAE.

## APIs
Check api.py for detail.  
Here is an example.  
```python
# import API and sanitize for I/O
from api import ScreenVAE_rec as SVAE
import lib.sanitize as sanitize

# Initialize an API object.
# You can choose to load other directories under checkpoints/ by passing their name as argument.
rec = SVAE()
# Read image. Friendly to Chinese paths.
img = sanitize.PILread('examples/manga.png')
line = sanitize.PILread('examples/line.png')
# Get screenmap with get_screenmap method.
scr = rec.get_screenmap(img, line)
# Get PCA image with get_pca method.
PCAimg = rec.get_pca(scr)
# Get recons image with get_recons method.
# This function is used internally in apply_screenmap method, and I don't know what it will do.
reconimg = rec.get_recons(scr)
# Get re-toned image with apply_screenmap method.
# Note that you need to give a seed point to decide where to be filled.
retoned = rec.apply_screenmap(img, scr, seedpt=(25, 75))
```