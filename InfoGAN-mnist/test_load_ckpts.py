# -*- coding: utf-8 -*-
#Use tensorflow to plot the variations using the checkpoints saved during training phase
#Checkpoint used to generate current images in repo are saved at epoch num - 60

import tensorflow as tf
import imageio
import natsort
import glob
import PIL
import IPython
import numpy as np
import matplotlib.pyplot as plt
from gan import Generator
from tensorflow_probability import distributions as tfd

generator = Generator()
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore("ckpt-60")

#creating sample noises to use it as control input 
def sample(z, size, cat=-1, c1=None, c2=None):
    if c1 is not None:
        z_con1 = np.array([c1] * size)
        z_con1 = np.reshape(z_con1, [size, 1])
    else:
        z_con1 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
        
    if c2 is not None:
        z_con2 = np.array([c2] * size)
        z_con2 = np.reshape(z_con2, [size, 1])
    else:
        z_con2 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
        
    if cat >= 0:
        z_cat = np.array([cat] * size)
        z_cat = tf.one_hot(z_cat, 10)
    else:
        z_cat = tfd.Categorical(probs=tf.ones([10])*0.1).sample([size,])
        z_cat = tf.one_hot(z_cat, 10)
    
    noise = tf.concat([z, z_con1, z_con2, z_cat], axis=-1)
    
    return noise

output_image = []
for i in range(10):
    z = tfd.Uniform(low=-1.0, high=1.0).sample([5, 62])
    noise = sample(z, 5, cat=i)
    imgs = generator(noise, training=False)
    imgs = (imgs + 1.) / 2.
    
    imgs = np.split(imgs, 5, 0)
    imgs = [np.reshape(img, [28, 28]) for img in imgs]
    imgs = np.concatenate(imgs, 0)
    output_image.append(imgs)
    
output_image = np.concatenate(output_image, 1)
output_image.shape

#plotting the variation of categorical variable
plt.figure(figsize=(12,8))
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.show()

output_image = []
cc = np.linspace(-2.0, 2.0, 10)
z = tfd.Uniform(low=-1.0, high=1.0).sample([1, 62])
for i in range(5):
    imgs = []
    for ii in range(10):
        noise = sample(z, 1, cat=i, c1=cc[ii], c2=0.0)
        img = generator(noise, training=False)[0]
        img = (img + 1.) / 2.
        imgs.append(np.reshape(img, [28, 28]))
    imgs = np.concatenate(imgs, 1)
    output_image.append(imgs)
    
output_image = np.concatenate(output_image, 0)
output_image.shape

#plotting the variation of 1st continuous variable
plt.figure(figsize=(12,8))
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.show()

output_image = []
cc = np.linspace(-2.0, 2.0, 10)
z = tfd.Uniform(low=-1.0, high=1.0).sample([1, 62])
for i in range(5):
    imgs = []
    for ii in range(10):
        noise = sample(z, 1, cat=i, c1=-1.0, c2=cc[ii])
        img = generator(noise, training=False)[0]
        img = (img + 1.) / 2.
        imgs.append(np.reshape(img, [28, 28]))
    imgs = np.concatenate(imgs, 1)
    output_image.append(imgs)
    
output_image = np.concatenate(output_image, 0)
output_image.shape

#plotting the variation of second continuous variable
plt.figure(figsize=(12,8))
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.show()
