import numpy as np
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.losses import mean_squared_error
from keras.layers import Lambda, Add, Activation, UpSampling2D, Dropout, LeakyReLU, BatchNormalization
import tensorflow as tf

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=None) # scaler

def bce_dice_coef(y_true, y_pred): # (batch_size, h, w)
    intersection = y_true * y_pred
    dice_c = 2.0 * K.sum(intersection)  / (K.sum(y_true) + K.sum(y_pred) + 1) # scaler
    bce = K.mean(K.binary_crossentropy(y_true, y_pred))
    loss = 0.5 * bce - dice_c + 1
    return loss

def conv(f, k=3, s=1, act=None):
    return Conv2D(f, (k, k), strides=(s,s), activation=act, kernel_initializer='he_normal', padding='same')

def model(input_shape=(224,224,3)):
    
    inputs = Input(shape=input_shape)
    
    x = conv(32, 3, 2) (inputs)  # 112x112x32
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = conv(32, 3, 2) (x)       # 56x56x32
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x) 
    
    x = conv(64, 3, 2) (x)       # 28x28x64
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    x = conv(64, 3, 2) (x)       # 14x14x64
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x) 
    
    x = conv(128, 3, 2) (x)      # 7x7x128
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x) 
    
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same') (x)
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same') (x)
    x = BatchNormalization() (x)    
    x = LeakyReLU(0.2) (x)
    
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same') (x)
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same') (x)
    x = BatchNormalization() (x)
    x = LeakyReLU(0.2) (x)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)
    
    return Model([inputs], [outputs])

if __name__ == '__main__':
    model().summary()