import theano
from theano import tensor as T
import numpy as np
from util import dnn_load_data, dnn_save_label, report_time, dnn_load_model
from dnn import DNN
import time, os

#---------- testing script ----------#

epoch         = 1200      # use [model_dir]/epoch.model 
batch_size    = 256
learning_rate = 0.05
lr_decay      = 1
dropout_prob  = [0.2, 0.2]
activation    = 'sigmoid'

feature = 'fbank4'
label_type = 'state'
data_size = '1M'

hidden = [2048, 2048]

parameters = '%s_%s_%s_nn%s_%s_epoch%d_lr%s_decay%s_drop%s' \
              %(feature, label_type, data_size, \
                "_".join(str(h) for h in hidden), activation, \
                epoch, str(learning_rate), str(lr_decay), \
                "_".join(str(p) for p in dropout_prob) )

model_dir = '../model/%s_%s_%s_nn%s_%s_lr%s_decay%s_drop%s' \
              %(feature, label_type, data_size, "_".join(str(h) for h in hidden), activation, \
                str(learning_rate), str(lr_decay), "_".join(str(p) for p in dropout_prob) )

# load model
model_filename = os.path.join(model_dir, 'epoch%d.model'%epoch)
dnn = dnn_load_model(model_filename)


# testing
test_filename = '../feature/test.%s' %feature
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s.csv' %parameters

pred_batch_size = 1000
Y_pred = dnn.batch_predict(X_test, pred_batch_size)
dnn_save_label('../frame/test.frame', output_filename, Y_pred, label_type)

