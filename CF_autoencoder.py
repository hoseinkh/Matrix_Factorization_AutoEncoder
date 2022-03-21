# =========================================================
# For more info, see https://hoseinkh.github.io/projects/
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz
#
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
## ********************************************************
## Hyperparameters:
batch_size = 128
epochs = 20
reg = 0.0001
## ********************************************************
new_train = load_npz("./Data/new_train.npz")
new_test = load_npz("./Data/new_test.npz")
### we also find mask variables to know which entries ...
# ... (of the User-Item rating matrix) are given.
# (recall that all the ratings are between 0.5 and 5).
mask_train = (new_train > 0) * 1.0
mask_test = (new_test > 0) * 1.0
## ****************************
# make copies since we will shuffle
new_train_copy = new_train.copy()
mask_train_copy = mask_train.copy()
new_test_copy = new_test.copy()
mask_test_copy = mask_test.copy()
#
#
N_train, M_train = new_train.shape
print("N_train:", N_train, "M_train:", M_train)
print("N_train // batch_size:", N_train // batch_size)
#
#
# center the data
mu_train = new_train.sum() / mask_train.sum()
print("mu_train:", mu_train)
#
#
## ********************************************************
def custom_loss(y_true, y_pred):
  # calculate the mse error for only those training entries ...
  # ... that users have rated befre (in the new_train data ).
  mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
  diff = y_pred - y_true
  sqdiff = diff * diff * mask
  sse = K.sum(K.sum(sqdiff))
  n = K.sum(K.sum(mask))
  return sse / n
## ********************************************************
def generator(A, M):
  # this function is specifically for the training
  # this function shuffles the data before each epoch, ...
  # ... and then groups the data into batches.
  while True:
    A, M = shuffle(A, M)
    for i in range(A.shape[0] // batch_size + 1):
      upper = min((i+1)*batch_size, A.shape[0])
      a = A[i*batch_size:upper].toarray()
      m = M[i*batch_size:upper].toarray()
      ## subtract the average only from those entries that have actual ratings.
      # ... this (on some rare occuasions can cause some problems. For instaince ...
      # ... if the mean mu_train is EXACTLY 3, then all the ratings that are 3 become ...
      # ... zero, which is not what we want. So we should be a little careful here!)
      a = a - mu_train * m # must keep zeros at zero!
      #
      if False:
        m2 = (np.random.random(a.shape) > 0.5)
        noisy = a * m2
      else: # (since we already are using DropOut)
        noisy = a # no noise
      yield noisy, a
## ********************************************************
def test_generator(A, M, A_test, M_test):
  # assumes A and A_test are in corresponding order
  # both of size N x M
  # here we do not need to shuffle, because we are "Testing" not "Training"
  while True:
    for i in range(A.shape[0] // batch_size + 1):
      upper = min((i+1)*batch_size, A.shape[0])
      a = A[i*batch_size:upper].toarray()
      m = M[i*batch_size:upper].toarray()
      at = A_test[i*batch_size:upper].toarray()
      mt = M_test[i*batch_size:upper].toarray()
      a = a - mu_train * m
      at = at - mu_train * mt
      yield a, at
## ********************************************************
#### build the model - just a 1 hidden layer autoencoder
i = Input(shape=(M_train,))
# bigger hidden layer size seems to help!
x = Dropout(0.7)(i)
x = Dense(700, activation='tanh', kernel_regularizer=l2(reg))(x)
# x = Dropout(0.5)(x)
x = Dense(M_train, kernel_regularizer=l2(reg))(x)
#
#
#### Create the model
model = Model(i, x)
model.compile(
  loss=custom_loss,
  optimizer=SGD(lr=0.08, momentum=0.9),
  # optimizer='adam',
  metrics=[custom_loss],
)
#
#
#### Fit the model
r = model.fit_generator(
  generator(new_train, mask_train),
  validation_data=test_generator(new_train_copy, mask_train_copy, new_test_copy, mask_test_copy),
  epochs=epochs,
  steps_per_epoch=new_train.shape[0] // batch_size + 1,
  validation_steps=new_test.shape[0] // batch_size + 1,
)
print(r.history.keys())
#
#
#
#### plot losses with regularization
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.xlabel("epoch")
plt.ylabel("MSE + Regularization Loss")
plt.legend()
plt.savefig("./figs/mse_and_regul_loss.png")
plt.show()
plt.close()
#
##### plot mse
plt.plot(r.history['custom_loss'], label="train mse")
plt.plot(r.history['val_custom_loss'], label="test mse")
plt.xlabel("epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("./figs/mse_error.png")
plt.show()
plt.close()

H = 5+6

