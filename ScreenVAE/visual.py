import pygubu, pathlib
from PIL import ImageTk, Image

import torch
import numpy as np
from api import ScreenVAE_rec as SVAE


PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "view_pattern.ui"
w, h = (768, 256)


def generate_gradient(cl1, cl2):
    out = np.zeros((w, 4))
    cl1 = np.array(cl1)[np.newaxis,:]
    cl2 = np.array(cl2)[np.newaxis,:]
    tmp = np.arange(w).astype(np.float32)[:,np.newaxis]/w
    out = tmp*(cl2-cl1)+cl1

    return out[np.newaxis,:,:].repeat(h+64, axis=0).astype(np.uint8)


class ViewPatternApp:
    def __init__(self, master=None):
        self.builder = builder = pygubu.Builder()
        builder.add_resource_path(PROJECT_PATH)
        builder.add_from_file(PROJECT_UI)
        # Main widget
        self.mainwindow = builder.get_object("top_window", master)
        builder.connect_callbacks(self)
        
        self.scale_objs = tuple(builder.get_object("scale%d"%n, master) for n in range(1, 8+1))
        self.img_label = builder.get_object("patt_img", master)
        patt = np.zeros((h, w))
        patt_tk = ImageTk.PhotoImage(Image.fromarray(patt))
        self.img_label.configure(image=patt_tk)
        self.img_label.image = patt_tk

    def run(self):
        self.mainwindow.mainloop()

    def gen_pattern(self, scale_value):
        scale_values = tuple(s.get() for s in self.scale_objs)
        cl1 = scale_values[0:4]; cl2 = scale_values[4:8]
        smap = generate_gradient(cl1, cl2).transpose(2,0,1)
        with torch.no_grad():
            patt = rec.map2img(smap)
        patt = patt[32:-32,:].astype(np.uint8)
        patt_tk = ImageTk.PhotoImage(Image.fromarray(patt))
        self.img_label.configure(image=patt_tk)
        self.img_label.image = patt_tk


if __name__ == "__main__":
    rec = SVAE(freeze_seed=0)
    app = ViewPatternApp()
    app.run()

