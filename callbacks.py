import os
import numpy as np
from keras.callbacks import Callback
from skimage.transform import resize
from skimage.io import imsave
class Preview(Callback):
    def __init__(self, preview_root, data_generator, base_model, **kwargs):
        super(Preview, self).__init__(**kwargs)
        self.preview_root = preview_root
        self.base_model = base_model # reference of model
        self.data_generator = data_generator
        if not os.path.exists(self.preview_root):
            os.makedirs(self.preview_root)
    def on_epoch_end(self, epoch, logs={}):
        total_n = len(self.data_generator)
        rand_idx = np.random.randint(total_n)
        x, y = self.data_generator.__getitem__(rand_idx)
        y_h = self.base_model.predict(x, batch_size=1)
        for n, xi, yi, y_hi in zip(range(len(x)), x, y, y_h):
            save_path = os.path.join(self.preview_root, 'epoch_{:d}_{:d}.jpg'.format(epoch, n))
            xi_h = resize(xi, yi.shape[:2])
            xi_h[np.squeeze(yi).astype(np.bool)] = 1.0, 0, 0
            img = np.round(np.concatenate([xi_h, np.tile(yi, (1,1,3)), np.tile(y_hi, (1,1,3)), np.tile(y_hi>.5, (1,1,3))], axis=1) * 255.0).astype(np.uint8)
            imsave(save_path, img)
        