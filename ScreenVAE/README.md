# ScreenVAE

Pytorch implementation for screentone encoding. 
For example, given the manga image with screentone, our model is able to generate feature representations which is plain within the region with same screentone and can also reconstruct the original manga image. 

**Note**: The current software works well with PyTorch 1.1+. 

## Example results


## Prerequisites
- Linux or Windows
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
python edit.py 
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
# It is recommended to freeze the seed for ramdom number to 0 to replicate the performance in paper.
rec = SVAE(freeze_seed=0)
# Read image. Friendly to Chinese paths.
img = sanitize.PILread('examples/manga.png')
line = sanitize.PILread('examples/line.png')
# Get screenmap with img2map method.
scr = rec.img2map(img, line)
# Convert screenmap to visual image with get_pca method.
PCAimg = rec.get_pca(scr)
# Convert screenmap back to screentone with map2img method.
retone = rec.map2img(scr)
# You may make use of lines like this:
retone = np.where(line<128, line, retone)
# You can get a quick view of result like this:
sanitize.PILshow(retone)
```