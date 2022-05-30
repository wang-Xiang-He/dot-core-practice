# 讀取 MNIST dataset 並進行前處理
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# x_train: Autoencoder 訓練資料
# x_test: Autoencoder 測試資料
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255. # 正規化資料數值範圍至 [0, 1] 間
x_test = x_test.astype('float32') / 255.

# 正規化資料維度，以便 Keras 處理
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

noise_factor = 0.5 # 決定 noise 的數量，值越大 noise 越多
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# 將資料限制在 [0, 1] 之間的範圍內
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# 使用加入 Noise 的資料建立 Autoencoder Model 並進行訓練

input_img = Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

# encoded size = (7, 7, 32) dimension
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x) 

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)

# Decoded size = (28, 28, 1) dimension (Original size)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# 訓練 DAE Model
autoencoder.fit(x_train_noisy, # 加入 noise 的資料為輸入
                x_train,  # 原始資料為 Label
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))