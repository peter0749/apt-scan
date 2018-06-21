import json
import math
import base64
import glob
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.utils import Sequence
import cv2
from sklearn.utils import shuffle
from skimage.transform import AffineTransform, warp
from skimage.draw import circle
from skimage.transform import rotate, resize
import copy

class APTDataset(Sequence):
    def __init__(self, prefix, input_shape, output_shape, batch_size=8, c_r=3.2, is_training=False):
        super().__init__()
        self.is_training = is_training
        self.c_r = c_r
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.json_files = glob.glob(prefix+'/**/*.json', recursive=True)
    def __len__(self):
        return int(np.ceil(float(len(self.json_files))/self.batch_size))

    def __getitem__(self, i):
        l_bound =  i    * self.batch_size
        r_bound = (i+1) * self.batch_size
        if r_bound>len(self.json_files): # ensure every iteration has the same batch size
            r_bound = len(self.json_files)
            l_bound = r_bound - self.batch_size
        dat_que = np.empty((self.batch_size, *self.input_shape),  dtype=np.float32)
        lab_que = np.empty((self.batch_size, *self.output_shape), dtype=np.float32)
        
        for n, index in enumerate(range(l_bound, r_bound)):
            with open(self.json_files[index]) as f:
                data = json.load(f)
            
            # Decode image from base64 imageData
            img = Image.open(BytesIO(base64.b64decode(data['imageData'])))
            img = img.convert('RGB')
            srcW, srcH = img.size
            dstW, dstH = self.output_shape[:2]
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, self.input_shape[:2][::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

            # Sort the corners by clockwise
            # while the first corner is the most top-lefted
            corners = np.float32(data['shapes'][0]['points'])
            corners[:, 0] = np.round(corners[:, 0] / srcW * dstW)  # x
            corners[:, 1] = np.round(corners[:, 1] / srcH * dstH)  # y
            
            lab = np.zeros(self.output_shape, dtype=np.float32)
            for (x, y) in corners:
                rr, cc = circle(y, x, self.c_r, shape=self.output_shape[:2])
                lab[rr, cc, 0] = 1 # markers
            
            if self.is_training:
                
                if np.random.rand() < .3: # flip vertical
                    img = np.flip(img, 0)
                    lab = np.flip(lab, 0)
                if np.random.rand() < .5: # flip horizontal
                    img = np.flip(img, 1)
                    lab = np.flip(lab, 1)
                if np.random.rand() < .3: # rotate
                    angle = np.random.uniform(-30,30)
                    img = rotate(img, angle, resize=True)
                    lab = rotate(lab, angle, resize=True)
                    if img.shape != self.input_shape:
                        img = resize(img, self.input_shape[:2],  mode='constant', cval=0, clip=True, preserve_range=True)
                        lab = resize(lab, self.output_shape[:2], mode='constant', cval=0, clip=True, preserve_range=True)
                
                # random amplify each channel
                a = .1 # amptitude
                t  = [np.random.uniform(-a,a)]
                t += [np.random.uniform(-a,a)]
                t += [np.random.uniform(-a,a)]
                t = np.array(t)

                img = np.clip(img * (1. + t), 0, 1) # channel wise amplify
                up = np.random.uniform(0.95, 1.05) # change gamma
                img = np.clip(img**up, 0, 1) # apply gamma and convert back to range [0,255]
            dat_que[n] = img
            lab_que[n] = lab
        return dat_que, (lab_que>.5).astype(np.uint8)

if __name__ == '__main__':
    data_generator = APTDataset('dataset', (224,224,3), (112,112,1), batch_size=4, is_training=True)
    batch = data_generator.__getitem__(0)
    plt.imshow(np.squeeze(np.round(batch[0][0]*255).astype(np.uint8)))
    plt.show()
    plt.imshow(np.squeeze(np.round(batch[1][0])))
    plt.show()