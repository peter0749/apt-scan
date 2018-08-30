import argparse
parser = argparse.ArgumentParser(description='AE-Apt-Scan')
parser.add_argument('--train', type=str, required=True,
                    help='path to training set')
parser.add_argument('--valid', type=str, required=True,
                    help='path to validation set')
parser.add_argument('--load_weights', action='store_true', default=False,
                    help='continue training')
parser.add_argument('--width', type=int, default=224, required=False,
                    help='width')
parser.add_argument('--height', type=int, default=224, required=False,
                    help='height')
parser.add_argument('--batch_size', type=int, default=8, required=False,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1000, required=False,
                    help='epochs')
parser.add_argument('--lr', type=float, default=0.001, required=False,
                    help='learning rate')
parser.add_argument('--early_stop', type=int, default=5, required=False,
                    help='early_stopping')
parser.add_argument('--loss_type', type=str, default='bce_dice', required=False,
                    help='bce_dice / focal_loss / bce / mse [default: bce_dice]')
parser.add_argument('--focal_loss_gamma', type=float, default=2.0, required=False,
                    help='focal_loss_gamma [default: 2.0]')
parser.add_argument('--focal_loss_alpha', type=float, default=0.3, required=False,
                    help='focal_loss_alpha [default: 0.2]')
parser.add_argument('--cpu_workers', type=int, default=5, required=False,
                    help='use how many process to load data?')
parser.add_argument('--queue_length', type=int, default=20, required=False,
                    help='data generator queue size')
parser.add_argument('--use_xla', action='store_true', default=False,
                    help='')
args = parser.parse_args()

print('Input size: {:d}x{:d}'.format(args.height, args.width))
print('Learning rate: {:.6f}'.format(args.lr))

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if args.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
session = tf.Session(config=config)
import keras
keras.backend.set_session(session)
from model import mean_iou, bce_dice_coef, model
from dataset import APTDataset
from callbacks import Preview
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from focal_loss import focal_loss

if args.loss_type=='bce_dice':
    loss = bce_dice_coef
elif args.loss_type=='focal_loss':
    loss = focal_loss(gamma=args.focal_loss_gamma, alpha=args.focal_loss_alpha)
elif args.loss_type=='bce':
    loss = 'binary_crossentropy'
else:
    loss = 'mean_squared_error'

ae = model(input_shape=(args.height, args.width, 3))
ae.compile(loss=loss, optimizer=Adam(lr=args.lr), metrics=[mean_iou])

if args.load_weights:
    ae.load_weights('best.h5')

ckpt = ModelCheckpoint('best.h5', save_best_only=True)

train_generator = APTDataset(args.train, (args.height, args.width, 3), (args.height//2, args.width//2, 1), batch_size=args.batch_size, is_training=True)
valid_generator = APTDataset(args.valid, (args.height, args.width, 3), (args.height//2, args.width//2, 1), batch_size=args.batch_size, is_training=False)

ae.fit_generator(train_generator, epochs=args.epochs, validation_data=valid_generator, shuffle=True, workers=args.cpu_workers, max_queue_size=args.queue_length, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=args.early_stop, mode='min', verbose=1), TensorBoard(), ckpt, Preview('./preview', valid_generator, ae)])
ae.save('ae.h5')
