import numpy as np
import random

train_num = 1000000
valid_num = 124823

train_filename = '../feature/train.fbank'
print "Load %s" %train_filename
X = np.loadtxt(train_filename, dtype='float')

(N, D) = X.shape
print "(N, D) = (%d, %d)" %(N, D)

sample_index = random.sample(range(N), train_num + valid_num)

train_index = sample_index[:train_num]
valid_index = sample_index[train_num+1:]

X_train = X[train_index, :]
X_valid = X[valid_index, :]

filename = '../feature/train1M.fbank'
print "Save %s" %filename
np.savetxt(filename, X_train, fmt='%.7f')

filename = '../feature/valid1M.fbank'
print "Save %s" %filename
np.savetxt(filename, X_valid, fmt='%.7f')

label_filename = '../label/train.48.index'
print "Load %s" %label_filename
label = np.loadtxt(label_filename, dtype='int')

label_train = label[train_index]
label_valid = label[valid_index]

filename = '../label/train1M.48.index'
print "Save %s" %filename
np.savetxt(filename, label_train, fmt='%d')

filename = '../label/valid1M.48.index'
print "Save %s" %filename
np.savetxt(filename, label_valid, fmt='%d')

