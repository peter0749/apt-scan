import argparse

parser = argparse.ArgumentParser(description='AE-Apt-Scan')
parser.add_argument('--test_dir', type=str, required=True,
                    help='path to test image')
parser.add_argument('--model', type=str, required=True,
                    help='path to model')
parser.add_argument('--component_threshold', type=int, default=10, required=False,
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
from unwarp import robust_unwarp, three_pts_to_four_pts, order_points
import json
import base64
import glob
from tqdm import tqdm
from PIL import Image
import cv2
from io import BytesIO

model = load_model(args.model, custom_objects={'bce_dice_coef':keras.losses.binary_crossentropy, 'focal_loss': keras.losses.binary_crossentropy, 'mean_iou': keras.losses.binary_crossentropy})
model.compile(loss='mse', optimizer='sgd')
ih, iw = model.input_shape[1:3]
oh, ow = model.output_shape[1:3]

test_data_paths = glob.glob(args.test_dir + '/*.json')
mean_sse_error = np.zeros(4, dtype=np.float32)
fails = 0
for path in tqdm(test_data_paths, total=len(test_data_paths)):
    with open(path) as f:
        data = json.load(f)
    img = Image.open(BytesIO(base64.b64decode(data['imageData'])))
    img = img.convert('RGB')
    srcW, srcH = img.size
    img = np.array(img, dtype=np.uint8)
    img = np.clip(cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0, 0, 1) # resize first... 
    true_corners = np.float32(data['shapes'][0]['points'])[:4] # format: (x,y)
    true_corners[:, 0] = np.clip(true_corners[:, 0] / float(srcW), 0, 1)
    true_corners[:, 1] = np.clip(true_corners[:, 1] / float(srcH), 0, 1)
    label = model.predict(img[np.newaxis,...], batch_size=1)[0]
    pts = postprocess.find_corner_condidate((label>0.5).astype(np.uint8), args.component_threshold) # format: (y, x)
    if len(pts)>0:
        pts[...,0] = np.clip(pts[...,0] / float(oh), 0, 1)
        pts[...,1] = np.clip(pts[...,1] / float(ow), 0, 1)
    if len(pts)==3:
        pts = three_pts_to_four_pts(pts)
    if len(pts)<4: # A failure case (cant wrap)
        pts = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.float32) # format: (y,x)
        fails += 1
    pts = order_points(pts[...,::-1]) # lu, ru, rd, ld. format: (x, y)
    true_corners = order_points(true_corners) # lu, ru, rd, ld. format: (x, y)
    sse_error = np.sum(np.square(true_corners - pts), axis=1)
    mean_sse_error += sse_error

mean_sse_error /= float(len(test_data_paths))
fail_r = fails / float(len(test_data_paths))
print('MSE: ', mean_sse_error)
print('mean MSE: ', mean_sse_error.mean())
print('failure rate: {:.6f}'.format(fail_r))