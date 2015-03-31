import numpy as np
import random


filename = 'train1M.list'
print "Load %s" %filename
train_index = np.loadtxt(filename, dtype='int')

filename = 'valid1M.list'
print "Load %s" %filename
valid_index = np.loadtxt(filename, dtype='int')

for fv in ['fbank', 'mfcc']:

    train_filename = '../feature/train.%s' %fv
    print "Load %s" %train_filename
    X = np.loadtxt(train_filename, dtype='float')

    X_train = X[train_index, :]
    X_valid = X[valid_index, :]

    filename = '../feature/train1M.%s' %fv
    print "Save %s" %filename
    np.savetxt(filename, X_train, fmt='%.7f')

    filename = '../feature/valid1M.%s' %fv
    print "Save %s" %filename
    np.savetxt(filename, X_valid, fmt='%.7f')


for label in ['48', 'state']:

    label_filename = '../label/train.%s.index' %label
    print "Load %s" %label_filename
    y = np.loadtxt(label_filename, dtype='int')

    y_train = y[train_index]
    y_valid = y[valid_index]

    filename = '../label/train1M.%s.index' %label
    print "Save %s" %filename
    np.savetxt(filename, y_train, fmt='%d')

    filename = '../label/valid1M.%s.index' %label
    print "Save %s" %filename
    np.savetxt(filename, y_valid, fmt='%d')

