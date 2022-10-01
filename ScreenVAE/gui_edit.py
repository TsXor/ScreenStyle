from api import ScreenVAE_rec as SVAE
import matplotlib.pyplot as plt
import lib.sanitize as sanitize

if __name__ == '__main__':
    rec = SVAE()
    img = sanitize.PILread('examples/manga.png')
    line = sanitize.PILread('examples/line.png')
    scr = rec.get_screenmap(img, line)
    resultPic = rec.get_pca(scr)

    def onclick(event):
        print(
            (
                '%s click: ' +
                'button=%d, ' +
                'x=%d, ' +
                'y=%d, ' +
                'xdata=%f, ' +
                'ydata=%f'
            ) % (
                'double' if event.dblclick else 'single',
                event.button,
                event.x,
                event.y,
                event.xdata,
                event.ydata
            )
        ) # Refresh the plot
        seedpt = (int(event.ydata),int(event.xdata))
        
        retoned = rec.apply_screenmap(img, scr, seedpt)
        
        plt.close(2)
        figi, axi = plt.subplots(figsize=(10, 8))
        axi.imshow(retoned, cmap=plt.cm.gray)
        axi.set_title('Edited'); axi.axis('off')
        figi.show()
        
        fig.canvas.draw()

    fig, ax = plt.subplots(ncols=2, figsize=(16, 8), dpi=100)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original'); ax[0].axis('off')
    ax[1].imshow(resultPic)
    ax[1].set_title('PCA'); ax[1].axis('off')
    
    plt.show()
