# Bidirectional Translation

Pytorch implementation for multimodal comic-to-manga translation. 

**Note**: The current software works well with PyTorch 1.6.0+. 

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone https://github.com/msxie/ScreenStyle.git
cd ScreenStyle/BidirectionalTranslation
```
- Install PyTorch and dependencies from http://pytorch.org
- ~~Install python libraries [tensorboardX](https://github.com/lanpa/tensorboardX)~~  
  This feature is currently unavailable.
- Install other libraries
For pip users:
```
pip install -r requirements.txt
```

## Data praperation
The training requires paired data (including manga image, western image and their line drawings). 
The line drawing can be extracted using [MangaLineExtraction](https://github.com/ljsabc/MangaLineExtraction).

  ```
${DATASET} 
|-- BidirectionalTranslation
|   |-- ${FOLDER}
|   |   |-- imgs
|   |   |   |-- 0001.png 
|   |   |   |-- ...
|   |   |-- line
|   |   |   |-- 0001.png 
|   |   |   |-- ...
  ```

### Use a Pre-trained Model
- Download the pre-trained [color2manga](https://drive.google.com/file/d/18-N1W0t3igWLJWFyplNZ5Fa2YHWASCZY/view?usp=sharing) model and place under `checkpoints/BidirectionalTranslation/` folder.
- Generate results with the model
```bash
python ./test.py
```

## Copyright and License
You are granted with the [LICENSE](../LICENSE) for both academic and commercial usages.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{xie-2020-manga,
        author   = {Minshan Xie and Chengze Li and Xueting Liu and Tien-Tsin Wong},
        title    = {Manga Filling Style Conversion with Screentone Variational Autoencoder},
        journal  = {ACM Transactions on Graphics (SIGGRAPH Asia 2020 issue)},
        month    = {December},
        year     = {2020},
        volume   = {39},
        number   = {6},
        pages    = {226:1--226:15}
    }
```

### Acknowledgements
This code borrows heavily from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.


## APIs
Check api.py for detail.  
Here is an example.  
```python
# Note: BidirectionalTranslation can only convert between ScreenVAE maps
# and colored images, so you need to use it with ScreenVAE.

# Manually include path to this project (the 'ScreenStyle' folder) in sys.path
import sys
sys.path.append('/path/to/this/project')

# import API
from ScreenVAE import SVAE
from BidirectionalTranslation import BT

# Initialize an API object.
# You can choose to load other directories under checkpoints/ by passing their name as argument.
# It is recommended to freeze the seed for ramdom number to 10 to replicate the performance in paper.
rec = SVAE(freeze_seed=0)
cvt = BT(freeze_seed=10)
# Get colored image with map2color method.
# You may use path (to a saved .npy file) or numpy array here.
scr = rec.img2map('examples/manga.png', 'examples/line.png')
color = cvt.map2color(scr, 'examples/line.png')
# Convert colored image back to screenmap with map2img method.
# You may use path (to an image) or numpy array here.
rescr = cvt.color2map(color, 'examples/line.png')
retone = rec.map2img(rescr)
```
Your can process multiple images **with the same shape** in one shot.  
But for batch functions you can only provide list of numpy arrays or a numpy array that is concatenated from image arrays.  
DO NOT TRY TO RESIZE MANGA IMAGES TO FIT THEM TO THE SAME SIZE because it will destroy screentones, but feel free to resize screenmaps or colored images because they are interpolative.  
```python
from PIL import Image
import numpy as np
from ScreenVAE import SVAE
from BidirectionalTranslation import BT

rec = SVAE(freeze_seed=0)
cvt = BT(freeze_seed=10)
img_paths = ['/path/to/your/image1', '/path/to/your/image2', ...]
imgs = [np.asarray(Image.open(p)) for p in img_paths]
line_paths = ['/path/to/your/line1', '/path/to/your/line2', ...]
lines = [np.asarray(Image.open(p)) for p in line_paths]
scrs = rec.img2map_batch(imgs, lines)
colors = cvt.map2color(scrs, lines)
```

## Training
I'm tired, really.