import theano
from theano import tensor as T
import numpy as np
from util import dnn_load_data, dnn_save_label, report_time, dnn_load_model
from dnn import DNN
import time, os

#---------- testing script ----------#

epoch         = 20      # use [model_dir]/epoch.model 
batch_size    = 100
learning_rate = 0.05
dropout_prob  = [0., 0.]

feature = 'fbank'
label_type = '48'

hidden = [128]

parameters = '%s_%s_nn%s_epoch%d_lr%s_drop%s' \
              %(feature, label_type, "_".join(str(h) for h in hidden), \
                epoch, str(learning_rate), \
                "_".join(str(p) for p in dropout_prob) )

model_dir = '../model/%s_%s_nn%s_lr%s_drop%s' \
              %(feature, label_type, "_".join(str(h) for h in hidden), \
                str(learning_rate), "_".join(str(p) for p in dropout_prob) )

# load model
model_filename = os.path.join(model_dir, 'epoch%d.model'%epoch)
dnn = dnn_load_model(model_filename)


# testing
test_filename = '../feature/test.fbank'
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s.csv' %parameters

pred_batch_size = 1000
Y_pred = dnn.batch_predict(X_test, pred_batch_size)
dnn_save_label('../frame/test.frame', output_filename, Y_pred, label_type)

