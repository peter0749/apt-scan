import argparse

parser = argparse.ArgumentParser(description='AE-Apt-Scan')
parser.add_argument('--image', type=str, required=True,
                    help='path to image')
parser.add_argument('--output', type=str, default='unwarpped.png', required=False,
                    help='path to output')
parser.add_argument('--model', type=str, required=True,
                    help='path to model')
parser.add_argument('--component_threshold', type=int, default=13, required=False,
                    help='path to output')
args = parser.parse_args()

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
keras.backend.set_session(session)
from keras.models import *
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.transform import resize
import postprocess
from unwarp import robust_unwarp

model = load_model(args.model, custom_objects={'bce_dice_coef':keras.losses.binary_crossentropy, 'focal_loss': keras.losses.binary_crossentropy, 'mean_iou': keras.losses.binary_crossentropy})
model.compile(loss='mse', optimizer='sgd')
ih, iw = model.input_shape[1:3]
oh, ow = model.output_shape[1:3]

img = imread(args.image)
img = gray2rgb(img)[...,:3]
img_r = resize(img, (ih, iw), preserve_range=True).astype(np.float32)[np.newaxis,...] / 255.0
h, w = img.shape[:2]

label = model.predict(img_r, batch_size=1)[0]

pts = postprocess.find_corner_condidate((label>0.5).astype(np.uint8), args.component_threshold) # format: (y, x)
if len(pts)==0:
    imsave(args.output, img)
    exit
pts[...,0] = np.clip(pts[...,0] * h / oh, 0, h-1)
pts[...,1] = np.clip(pts[...,1] * w / ow, 0, w-1)
w = robust_unwarp(img, pts)

imsave(args.output, w)

