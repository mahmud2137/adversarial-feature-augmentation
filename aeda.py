import pickle
from tensorflow import keras
### hack tf-keras to appear as top level keras

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

# import sys
# sys.modules['keras'] = keras
from keras.layers import *
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import scipy.io
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_svhn(image_dir, split='train'):
    print('Loading SVHN dataset.')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_mnist(image_dir, split='train'):
    print('Loading MNIST dataset.')

    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']

    return images, np.squeeze(labels).astype(int)

class AutoEncoder_MNIST():
    def __init__(self, input_shape, bottle_neck_shape = 16*16):
        self.input_shape = input_shape
        self.bottle_neck_shape = bottle_neck_shape
        self.build_model()

    def encoder_layers(self,x):
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), input_shape = self.input_shape, activation = 'relu')(x)
        x = MaxPooling2D(pool_size = (2,2))(x)

        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPooling2D(pool_size = (2,2))(x)
        x = Flatten()(x)
        # x = Dense(50, activation='relu')(x)
        encoded = Dense(self.bottle_neck_shape, activation='relu')(x)

        return encoded

    def decoder_layers(self, x):
        # x = Dense(50, activation='relu')(x)
        x = Dense(8*8*32, activation='relu')(x)
        x = Reshape((8, 8, 32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        decoded = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear')(x)

        return decoded

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        self.bottle_neck_layer = self.encoder_layers(input_layer)
        output_layer = self.decoder_layers(self.bottle_neck_layer)
        self.encoder = Model(input_layer, self.bottle_neck_layer)
        self.autoencoder = Model(input_layer, output_layer)
        
if __name__ == "__main__":
    mnist_dir='./data/mnist'
    data_mode = 'test'
    mnist_img, mnist_label = load_mnist(mnist_dir, split= data_mode)
    mnist_img.shape[1:]
    ae_mnist = AutoEncoder_MNIST(mnist_img.shape[1:])
    ae_mnist.build_model()
    ae_mnist.encoder.summary()

    opt = Adam(lr=0.000001)
    ae_mnist.autoencoder.compile(loss = 'mse',
            optimizer = opt)
            
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    if data_mode == 'train':
        hist = ae_mnist.autoencoder.fit(mnist_img, mnist_img,
                        epochs=1000,
                        batch_size=64,
                        shuffle=True,
                        verbose=2,
                        callbacks=[stop_early],
                        validation_split = 0.15)
        
        ae_mnist.encoder.save_weights("ae_models/mnist_encoder.h5")
        plt.figure(figsize=(10,8))
        plt.plot(hist.history['loss'], label = 'Training Loss')
        plt.plot(hist.history['val_loss'], label = 'Validation Loss')
        plt.legend()
        plt.savefig("mnist_ae_learning_curve.png")
        plt.show()

    ae_mnist.encoder.load_weights("ae_models/mnist_encoder.h5")
    enc_img_mnist = ae_mnist.encoder.predict(mnist_img)
    enc_img_mnist = enc_img_mnist.reshape(-1,16,16,1)
    enc_img_mnist_dict = {'X': enc_img_mnist, 'y': mnist_label}
    if data_mode == 'train':
        enc_data_path = "data_encoded/mnist/train.pkl"
    else:
        enc_data_path = "data_encoded/mnist/test.pkl"
    with open( enc_data_path , 'wb') as enc_file:
        pickle.dump(enc_img_mnist_dict, enc_file)

    ############################################
    #Loading and Encoder for SVHN data
    ############################################
    svhn_dir= './data/svhn'
    svhn_img, svhn_label  = load_svhn(svhn_dir, split=data_mode)
    svhn_img.shape[3]
    ae_svhn = AutoEncoder_MNIST(svhn_img.shape[1:])
    ae_svhn.build_model()
    ae_svhn.autoencoder.summary()

    opt = Adam(lr=0.000001)
    ae_svhn.autoencoder.compile(loss = 'mse',
            optimizer = opt)
            
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    if data_mode == 'train':
        hist = ae_svhn.autoencoder.fit(svhn_img, svhn_img,
                        epochs=500,
                        batch_size=64,
                        shuffle=True,
                        verbose=2,
                        callbacks=[stop_early],
                        validation_split = 0.15)
        ae_svhn.encoder.save_weights("ae_models/svhn_encoder.h5")

        plt.figure(figsize=(10,8))
        plt.plot(hist.history['loss'], label = 'Training Loss')
        plt.plot(hist.history['val_loss'], label = 'Validation Loss')
        plt.legend()
        plt.savefig("svhn_ae_learning_curve.png")
        plt.show()

    ae_svhn.encoder.load_weights("ae_models/svhn_encoder.h5")

    enc_img_svhn = ae_svhn.encoder.predict(svhn_img)
    enc_img_svhn = enc_img_svhn.reshape(-1,16,16,1)
    enc_img_svhn_dict = {'X': enc_img_svhn, 'y': svhn_label}
    if data_mode == 'train':
        enc_file_path = "data_encoded/svhn/train.pkl"
    else:
        enc_file_path = "data_encoded/svhn/test.pkl"

    with open( enc_file_path , 'wb') as enc_file:
        pickle.dump(enc_img_svhn_dict, enc_file)

