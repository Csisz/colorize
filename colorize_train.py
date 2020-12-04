import matplotlib.pyplot as plt
import cv2
import numpy as np 
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape,LeakyReLU, Dropout
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D,UpSampling2D
from tensorflow import keras

ab = np.load('ab1.npy')
gray = np.load('gray_scale.npy')

def batch_prep (gray_img,batch_size=100):
    img=np.zeros((batch_size,224,224,3))
    for i in range (0,3):
      img[:batch_size,:,:,i]=gray_img[:batch_size]
      return img
img_in=batch_prep(gray,batch_size=300)

def get_rbg(gray_imgs,ab_imgs,n=10):
    img1=np.zeros((n,224,224,3))
    img1[:,:,:,0]=gray_imgs[0:n:]
    img1[:,:,:,1:]=ab_imgs[0:n]
    img1=img1.astype('uint8')
    imgs=[]
    for i in range(0,n):
        imgs.append(cv2.cvtColor(img1[i],cv2.COLOR_LAB2RGB))
        
    imgs=np.array(imgs)
    
    return imgs

img_out = get_rbg(gray_imgs = gray, ab_imgs = ab, n = 300)

model = Sequential()
model.add(Conv2D(strides=1,kernel_size=3,filters=12,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(Conv2DTranspose(strides=1,kernel_size=3,filters=12,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(LeakyReLU(0.6))
model.add(Dropout(0.4))

model.add(Conv2D(strides=1,kernel_size=3,filters=12,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(Conv2DTranspose(strides=1,kernel_size=3,filters=3,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(LeakyReLU(0.6))
model.add(Dropout(0.4))

model.add(Conv2D(strides=1,kernel_size=3,filters=12,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(Conv2DTranspose(strides=1,kernel_size=3,filters=3,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(LeakyReLU(0.6))
model.add(Dropout(0.4))

model.add(Conv2D(strides=1,kernel_size=3,filters=12,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(Conv2DTranspose(strides=1,kernel_size=3,filters=3,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(UpSampling2D((2,2)))
model.add(LeakyReLU(0.6))
model.add(Dropout(0.4))

model.add(Conv2D(strides=1,kernel_size=3,filters=12,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(Conv2DTranspose(strides=1,kernel_size=3,filters=3,use_bias=True,bias_initializer=tf.keras.initializers.RandomUniform(minval=0.05,maxval=0.05),padding="valid",activation=tf.nn.relu))
model.add(LeakyReLU(0.6))
model.add(Dropout(0.4))

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mape',metrics=tf.keras.metrics.Accuracy())
model.fit(img_in,img_out,epochs=10,batch_size=16)
prediction=model.predict(img_in)

model.save('model_color.h5')
# plt.imshow(prediction[29])
# plt.show()

