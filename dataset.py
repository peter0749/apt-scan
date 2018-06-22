import json
import math
import base64
import glob
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm
from keras.utils import Sequence
import cv2
from sklearn.utils import shuffle
from skimage.transform import AffineTransform, warp
from skimage.draw import circle
from skimage.transform import rotate, resize
import copy

R_t = lambda theta: np.array([[math.cos(theta), -math.sin(theta)],
                              [math.sin(theta),  math.cos(theta)]], dtype=np.float32)

class APTDataset(Sequence):
    def __init__(self, prefix, input_shape, output_shape, batch_size=8, c_r=3.2, is_training=False):
        super().__init__()
        self.is_training = is_training
        self.c_r = c_r
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.json_files = glob.glob(prefix+'/**/*.json', recursive=True)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.seq = iaa.Sequential(
            [
                # execute 0 to 3 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 3),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 0.3), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.Multiply((0.8, 1.2), per_channel=0.5),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0))
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
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
            
            crop_ratio = np.zeros(4, dtype=np.float32)
            
            if self.is_training and np.random.rand() < 0.3:
                crop_ratio = np.random.uniform(0.01, 0.1, size=4)
                u, r, d, l = np.round(crop_ratio * np.array([srcH, srcW, srcH, srcW])).astype(np.uint8)
                img = img[u:srcH-d,l:srcW-r] # crop image
            fx = self.input_shape[1] / float(img.shape[1])
            fy = self.input_shape[0] / float(img.shape[0])
            
            img = cv2.resize(img, self.input_shape[:2][::-1], interpolation=cv2.INTER_AREA) # resize first... 

            # Sort the corners by clockwise
            # while the first corner is the most top-lefted
            corners = np.float32(data['shapes'][0]['points'])
            
            if self.is_training and np.sum(crop_ratio)>0:
                corners[:, 0] -= l 
                corners[:, 1] -= u 
            
            corners[:, 0] *= fx
            corners[:, 1] *= fy
            
            if self.is_training and np.random.rand() < .3:
                angle = np.random.uniform(-30,30)
                cx = int(img.shape[1]//2)
                cy = int(img.shape[0]//2)
                
                M = cv2.getRotationMatrix2D((cx,cy),angle,1)
                
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
 
                (h, w) = img.shape[:2]
                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
 
                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cx
                M[1, 2] += (nH / 2) - cy
                
                img = np.clip(cv2.warpAffine(img,M,(nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.random.randint(25)), 0, 255)
                
                x_scale = self.input_shape[1] / nW
                y_scale = self.input_shape[0] / nH
                
                if img.shape != self.input_shape:
                    img = cv2.resize(img, self.input_shape[:2][::-1], interpolation=cv2.INTER_AREA) # resize first... 
                    # img = resize(img, self.input_shape[:2],  mode='constant', cval=0, clip=True, preserve_range=True, order=0)
                
                R = R_t(-angle*np.pi/180.0)
                corners[:, 0] -= cx
                corners[:, 1] -= cy
                corners = (R @ corners.T).T
                corners[:, 0] *= x_scale
                corners[:, 1] *= y_scale
                corners[:, 0] += cx
                corners[:, 1] += cy
            
            corners[:, 0] = np.round(np.clip(corners[:, 0] * dstW/self.input_shape[1], 0, dstW-1))
            corners[:, 1] = np.round(np.clip(corners[:, 1] * dstH/self.input_shape[0], 0, dstH-1))
            corners = corners.astype(np.uint8)
            lab = np.zeros(self.output_shape, dtype=np.float32)
            for (x, y) in corners:
                rr, cc = circle(y, x, self.c_r, shape=self.output_shape[:2])
                lab[rr, cc, 0] = 1 # markers
            
            if self.is_training:
                if np.random.rand() < 0.3: # heavy augmentation (slow)
                    img = self.seq.augment_image(img) # data augmentation
                else: # light augmentation (fast)
                    img = img.astype(np.float32) / 255.0 # normalize first
                    # random amplify each channel
                    a = .1 # amptitude
                    t  = [np.random.uniform(-a,a)]
                    t += [np.random.uniform(-a,a)]
                    t += [np.random.uniform(-a,a)]
                    t = np.array(t)

                    img = np.clip(img * (1. + t), 0, 1) # channel wise amplify
                    up = np.random.uniform(0.95, 1.05) # change gamma
                    img = np.clip(img**up, 0, 1) # apply gamma and convert back to range [0,255]    
                    
                    # additive random noise
                    sigma = np.random.rand()*0.04
                    img = np.clip(img + np.random.randn(*img.shape)*sigma, 0, 1)
                    
                    img = np.round(np.clip(img*255, 0, 255)).astype(np.uint8)
                                        
                    if np.random.binomial(1, .05):
                        ksize = np.random.choice([3,5,7])
                        img = cv2.GaussianBlur(img, (ksize,ksize), 0)
            
            img = np.clip(img.astype(np.float32) / 255.0, 0, 1) # normalize
            if self.is_training:
                if np.random.rand() < .3: # flip vertical
                    img = np.flip(img, 0)
                    lab = np.flip(lab, 0)
                if np.random.rand() < .5: # flip horizontal
                    img = np.flip(img, 1)
                    lab = np.flip(lab, 1)
            dat_que[n] = img
            lab_que[n] = lab
        return dat_que, (lab_que>.5).astype(np.uint8)

if __name__ == '__main__':
    data_generator = APTDataset('dataset\\valid', (224,224,3), (112,112,1), batch_size=4, is_training=True)
    batch = data_generator.__getitem__(0)
    img = np.round(np.clip(batch[0][0]*255, 0, 255)).astype(np.uint8)
    lab = np.squeeze(batch[1][0]).astype(np.bool)
    img = resize(img, (112,112))
    plt.imshow(img)
    plt.show()
    plt.imshow(lab)
    plt.show()
    img[lab] = 255, 0, 0
    plt.imshow(img)
    plt.show()