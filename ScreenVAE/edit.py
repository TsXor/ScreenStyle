from api import ScreenVAE_rec as SVAE
import matplotlib.pyplot as plt
import tkinter.filedialog as tkfd
import pathlib
import lib.sanitize as sanitize
import numpy as np
from skimage.segmentation import flood, flood_fill


def fillin_smap(img, scr, seedpt=(10,10)):
    filled = np.ones(scr.shape[1:])
    # scr[line[np.newaxis,:,:].repeat(4,axis=0)<0.75]=-1
    nscr = scr.copy()
    for i in range(4):
        filled_img = flood(nscr[i], seedpt, tolerance=0.15)
        filled[~filled_img] = 0
    return filled


if __name__ == '__main__':
    imdir = tkfd.askdirectory(title='choose input directory of pictures')
    imdir = imdir if imdir else 'examples' # default to examples if cancelled
    imdir = pathlib.Path(imdir)
    rec = SVAE(freeze_seed=0)
    img = sanitize.PILread(imdir/'manga.png')
    line = sanitize.PILread(imdir/'line.png')
    scr = rec.img2map(img, line)
    smap_visualize = rec.get_pca(scr)
    np.save(imdir/'scrmap.npy', scr)
    out = rec.map2img(scr)
    sanitize.PILshow(np.where(line<128, line, out))

    def onclick(event):
        eventname = 'double' if event.dblclick else 'single'
        if (event.ydata is None) or (event.xdata is None):
            print('not clicking in any picture, skipping..')
            return
        seedpt = (int(event.ydata),int(event.xdata))
        
        print(
            '%s click: '  % eventname    +
            'button=%d, ' % event.button +
            'x=%d, '      % event.x      +
            'y=%d, '      % event.y      +
            'xdata=%f, '  % event.xdata  +
            'ydata=%f'    % event.ydata
        )
        
        mask = fillin_smap(img, scr, seedpt)
        result = np.where(mask, out, img)
        
        # Refresh the plot
        plt.close(2)
        figi, axi = plt.subplots(figsize=(10, 8))
        axi.imshow(result, cmap=plt.cm.gray)
        axi.set_title('Edited'); axi.axis('off')
        figi.show()
        
        fig.canvas.draw()

    fig, ax = plt.subplots(ncols=3, figsize=(24, 8), dpi=100)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original'); ax[0].axis('off')
    ax[1].imshow(smap_visualize)
    ax[1].set_title('PCA'); ax[1].axis('off')
    ax[2].imshow(out)
    ax[2].set_title('Re-toned'); ax[2].axis('off')
    
    plt.show()
