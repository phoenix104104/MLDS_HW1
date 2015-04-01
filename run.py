import theano
from theano import tensor as T
import numpy as np
from util import dnn_load_data, dnn_save_label, report_time
from dnn import DNN
from pickle import dump, load
import time


epoch         = 200
batch_size    = 100
learning_rate = 0.05
dropout_prob  = 0.

feature = 'fbank'
label_type = '48'
N_class = 48

data_size = '1M'
train_filename = '../feature/train%s.%s' %(data_size, feature)
train_labelname = '../label/train%s.%s.index' %(data_size, label_type)
print "Load %s" %train_filename
X_train, Y_train = dnn_load_data(train_filename, train_labelname, N_class)

valid_filename = '../feature/valid%s.%s' %(data_size, feature)
valid_labelname =  '../label/valid%s.%s.index' %(data_size, label_type)
print "Load %s" %valid_filename
X_valid, Y_valid = dnn_load_data(valid_filename, valid_labelname, N_class)

(N_data, N_dim) = X_train.shape

structure = [N_dim, 2048, 2048, N_class]

# load model
'''
model_filename = 'test.model'
with open(model_filename, 'r') as file:
    print "Load model %s" %model_filename
    dnn = load(file)
'''

# training
print "Start DNN training..."
print "NN structure: %s" %("-".join(str(s) for s in structure))
print "Learning rate = %f, epoch = %d" %(learning_rate, epoch)
print "Dropout probability = %s" %(str(dropout_prob))
dnn = DNN(structure, learning_rate, epoch, batch_size, dropout_prob)

ts = time.time()
acc_all = dnn.train(X_train, Y_train, X_valid, Y_valid)
te = time.time()
report_time(ts, te)


# save model

parameters = '%s_%s_nn%s_epoch%d_lr%s_drop%s' \
              %(feature, label_type, "_".join(str(h) for h in structure), \
                epoch, str(learning_rate), str(dropout_prob) )

model_filename = '../model/%s.model' %parameters
with open(model_filename, 'w') as file:
    print "Save model %s" %model_filename
    dump(dnn, file)

# save accuracy log
log_filename = '../log/%s.log' %parameters
print "Save %s" %log_filename
np.savetxt(log_filename, acc_all, fmt='%.7f')


# clear X_train, X_valid
X_train = []
Y_train = []
X_valid = []
Y_valid = []


# testing
test_filename = '../feature/test.fbank'
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s.csv' %parameters

pred_batch_size = 1000
Y_pred = dnn.batch_predict(X_test, pred_batch_size)
dnn_save_label('../frame/test.frame', output_filename, Y_pred, label_type)

