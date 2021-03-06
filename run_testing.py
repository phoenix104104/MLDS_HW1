import theano
from theano import tensor as T
import numpy as np
from util import OPTS, dnn_load_data, dnn_save_label, report_time, dnn_load_model
from dnn import DNN
import time, os

#---------- testing script ----------#

epoch = 500      # use [model_dir]/epoch.model 

opts = OPTS()
opts.learning_rate  = 0.01
opts.lr_decay       = 1
opts.momentum       = 0.9
opts.weight_decay   = 0.0005
opts.rmsprop_alpha  = 0.9
opts.dropout_prob   = 0.
#opts.activation     = 'sigmoid'
#opts.activation     = 'tanh'
opts.activation     = 'ReLU'
opts.update_grad    = 'sgd'
#opts.update_grad    = 'rmsprop' # use learning-rate = 0.001

opts.hidden = [2048, 2048]

opts.data_size = '1M'
opts.feature = 'fbank8.norm'
opts.label_type = '48'
if( opts.label_type == '48' ):
    opts.N_class = 48
elif( opts.label_type == 'state' ):
    opts.N_class = 1943

parameters = '%s_%s_nn%s_%s_%s_lr%s_ld%s_m%s_wd%s_drop%s_%s_alpha%s' \
              %(opts.feature, opts.data_size, \
                "_".join(str(h) for h in opts.hidden), \
                opts.label_type, \
                opts.activation, \
                str(opts.learning_rate), \
                str(opts.lr_decay), \
                str(opts.momentum), \
                str(opts.weight_decay), \
                str(opts.dropout_prob), \
                opts.update_grad, \
                str(opts.rmsprop_alpha) )

opts.model_dir = '../model/%s' %parameters

# load model
model_filename = os.path.join(opts.model_dir, 'epoch%d.model'%epoch)
dnn = dnn_load_model(model_filename)


# testing (old data)
test_filename = '../feature/test.old.%s' %opts.feature
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s_epoch%d.old.csv' %(parameters, epoch)

Y_pred = dnn.predict(X_test)
dnn_save_label('../frame/test.old.frame', output_filename, Y_pred, opts.label_type)



# testing (final data)
test_filename = '../feature/test.%s' %opts.feature
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s_epoch%d.csv' %(parameters, epoch)

Y_pred = dnn.predict(X_test)
dnn_save_label('../frame/test.frame', output_filename, Y_pred, opts.label_type)


