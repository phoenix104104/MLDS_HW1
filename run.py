import theano
from theano import tensor as T
import numpy as np
from util import dnn_load_data
from dnn import DNN
from pickle import dump, load


epoch = 100
batch_size = 100
learning_rate = 0.05

N_class = 48
X_train, Y_train = dnn_load_data('../feature/train1M.fbank', '../label/train1M.48.index', N_class)
X_valid, Y_valid = dnn_load_data('../feature/valid1M.fbank', '../label/valid1M.48.index', N_class)

(N_data, N_dim) = X_train.shape

structure = [N_dim, 128, N_class]

# load model
'''
model_filename = 'test.model'
with open(model_filename, 'r') as file:
    print "Load model %s" %model_filename
    dnn = load(file)
'''

# training
dnn = DNN(structure, learning_rate, epoch, batch_size)
dnn.train(X_train, Y_train, X_valid, Y_valid)


# save model
'''
model_filename = 'test.model'
with open(model_filename, 'w') as file:
    print "Save model %s" %model_filename
    dump(dnn, file)
'''

