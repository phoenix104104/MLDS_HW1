import theano
from theano import tensor as T
import numpy as np
from util import dnn_load_data
from dnn import DNN
from pickle import dump, load


epoch         = 100
batch_size    = 100
learning_rate = 0.01
dropout_prob  = 0.5

N_class = 48

data_size = '1M'
train_filename = '../feature/train%s.fbank' %data_size
train_labelname = '../label/train%s.48.index' %data_size
print "Load %s" %train_filename
X_train, Y_train = dnn_load_data(train_filename, train_labelname, N_class)

valid_filename = '../feature/valid%s.fbank' %data_size
valid_labelname =  '../label/valid%s.48.index' %data_size
print "Load %s" %valid_filename
X_valid, Y_valid = dnn_load_data(valid_filename, valid_labelname, N_class)

(N_data, N_dim) = X_train.shape

structure = [N_dim, 1000, 1000, N_class]

# load model
'''
model_filename = 'test.model'
with open(model_filename, 'r') as file:
    print "Load model %s" %model_filename
    dnn = load(file)
'''

# training
print "Start DNN training..."
dnn = DNN(structure, learning_rate, epoch, batch_size, dropout_prob)
dnn.train(X_train, Y_train, X_valid, Y_valid)


# save model
'''
model_filename = 'test.model'
with open(model_filename, 'w') as file:
    print "Save model %s" %model_filename
    dump(dnn, file)
'''

