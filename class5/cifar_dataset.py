import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

cifars=tf.keras.datasets.cifar10

(xtrain,ytrain),(xtest,ytest)=cifars.load_data()
print(ytrain[0])
plt.imshow(xtrain[0])
plt.show()
