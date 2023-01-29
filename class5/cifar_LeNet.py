import os
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Conv2D, BatchNormalization,Activation,Dropout,Flatten,Dense
from tensorflow.keras.models import Model, Sequential

import numpy as np
# from matplotlib import pyplot as plt

cifars=tf.keras.datasets.cifar10

(xtrain,ytrain),(xtest,ytest)=cifars.load_data()
xtrain, xtest=xtrain/255.0,xtest/255.0
class BaseLine(Model):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.c1=Conv2D(filters=6,kernel_size=(5,5),padding='same', activation='sigmoid')#c
        self.p1=MaxPool2D(pool_size=(2,2), strides=2)

        self.c2=Conv2D(filters=16, kernel_size=(5,5), activation='sigmoid')
        self.p2=MaxPool2D(pool_size=(2,2), strides=2)

        self.flatten=Flatten()
        self.f1=Dense(128,activation='sigmoid')
        self.f2=Dense(84, activation='sigmoid')
        self.f3=Dense(10, activation='sigmoid')


    def call(self, x):
        x=self.c1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.p2(x)

        x=self.flatten(x)
        x=self.f1(x)
        x=self.f2(x)
        y=self.f3(x)

        return y

model = BaseLine()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
    )
check_point_path = "./checkpoint/baseline.ckpt"
if os.path.exists(check_point_path + ".index"):
    model.load_weights(check_point_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path,
                                               save_weights_only=True,
                                               save_best_only=True
                                                )
history=model.fit(
    xtrain,ytrain,
    batch_size=32,
    epochs=5,
    validation_data=(xtest,ytest),
    validation_freq=1,
    callbacks=[cp_callback]
    )
model.summary()
with open('./weights.txt', 'w') as f:
    for v in model.trainable_variables:
        f.write(str(v.name)+"\n")
        f.write(str(v.shape)+"\n")
        f.write(str(v.numpy())+"\n")

