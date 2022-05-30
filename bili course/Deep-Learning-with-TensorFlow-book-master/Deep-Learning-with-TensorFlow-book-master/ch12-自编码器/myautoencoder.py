# 讀取 MNIST dataset 並進行前處理
from keras.datasets import mnist
import numpy as np

# x_train: Autoencoder 訓練資料
# x_test: Autoencoder 測試資料
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255. # 正規化資料數值範圍至 [0, 1] 間
x_test = x_test.astype('float32') / 255.

# 正規化資料維度，以便 Keras 處理
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# 資料讀取完成後，接著建立 Autoencoder Model 並使用 x_train 資料進行訓練。

from keras.layers import Input, Dense
from keras.models import Model

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img) ## Encoding layer 設為 32 維
decoded = Dense(784, activation='sigmoid')(encoded) ## Decoding layer 設為與 input layer 相同的 784 維

# 建立 Model 並將 loss funciton 設為 binary cross entropy
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train,
                x_train,  # Label 也設為 x_train
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))