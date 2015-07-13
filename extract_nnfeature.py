import theano
from theano import tensor as T
import numpy as np
from util import OPTS, dnn_load_data, dnn_save_label, report_time, dnn_save_model, dnn_load_model, dnn_save_data
from dnn import DNN
from pickle import dump, load
import time, sys, os

sys.setrecursionlimit(9999) # to dump large network
#---------- training script ----------#

opts = OPTS()
opts.epoch          = 1000
opts.batch_size     = 256
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
opts.pretrain       = 0

opts.hidden         = [2048, 2048]
opts.data_size      = 'all'
opts.feature        = 'fbank1.norm'
opts.label_type     = '48'

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
model_filename = os.path.join(opts.model_dir, 'epoch%d.model'%opts.epoch)
dnn = dnn_load_model(model_filename)


# output
layer = 3
fv_out = '%s.%s_nn%s_%s_drop%s.L%d' \
         %(opts.feature, \
           opts.data_size, \
           "_".join( str(h) for h in opts.hidden), \
           opts.label_type, \
           opts.dropout_prob, \
           layer)

output_dir = '../../hw2/hw1_feature'


for t in ["train", "test", "test.old"]:

    filename = '../feature/%s.%s' %(t, opts.feature)
    X = dnn_load_data(filename)

    print "Extract dnn feature..."
    feature = dnn.get_hidden_feature(X, layer)

    output_filename = os.path.join(output_dir, '%s.%s' %(t, fv_out))
    dnn_save_data(output_filename, feature)

