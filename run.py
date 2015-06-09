import theano
from theano import tensor as T
import numpy as np
from util import OPTS, dnn_load_data, dnn_save_label, report_time, dnn_save_model, dnn_load_model
from dnn import DNN
from pickle import dump, load
import time, sys, os


sys.setrecursionlimit(9999) # to dump large network
#---------- training script ----------#

opts = OPTS()
opts.epoch          = 100
opts.batch_size     = 256
opts.learning_rate  = 0.001
opts.lr_decay       = 1
opts.momentum       = 0.9
opts.weight_decay   = 0.0005
opts.rmsprop_alpha  = 0.9
opts.dropout_prob   = 0.0
opts.activation     = 'sigmoid'
#opts.activation     = 'tanh'
#opts.activation     = 'ReLU'
#opts.update_grad    = 'sgd'
opts.update_grad    = 'rmsprop' # use learning-rate = 0.001

opts.hidden = [100, 100]

opts.data_size = '1M'
opts.feature = 'fbank'
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
print "Save model at %s" %opts.model_dir

if( os.path.isdir(opts.model_dir) ):
    print "Warning! Directory %s already exists!\n" %opts.model_dir
    print ">>>>> ctrl+c to leave the program, or press any key to continue..."
    raw_input()
else:
    print "mkdir %s" %opts.model_dir
    os.mkdir(opts.model_dir)

if( opts.data_size == 'all' ):
    train_filename = '../feature/train.%s' %(opts.feature)
    train_labelname = '../label/train.%s.index' %(opts.label_type)
else:
    train_filename = '../feature/train%s.%s' %(opts.data_size, opts.feature)
    train_labelname = '../label/train%s.%s.index' %(opts.data_size, opts.label_type)

X_train, Y_train = dnn_load_data(train_filename, train_labelname, opts.N_class)

if( opts.data_size == 'all' ):
    valid_filename = '../feature/valid1M.%s' %(opts.feature)
    valid_labelname =  '../label/valid1M.%s.index' %(opts.label_type)
else:
    valid_filename = '../feature/valid%s.%s' %(opts.data_size, opts.feature)
    valid_labelname =  '../label/valid%s.%s.index' %(opts.data_size, opts.label_type)

X_valid, Y_valid = dnn_load_data(valid_filename, valid_labelname, opts.N_class)

(N_data, N_dim) = X_train.shape

opts.structure = [N_dim] + opts.hidden + [opts.N_class]


print "Build DNN structure..."
dnn = DNN(opts)


# training
print "Start DNN training...(lr_decay = %s)" %(str(opts.lr_decay))

acc_all = []
lr = opts.learning_rate
ts = time.time()
for i in range(opts.epoch):

    dnn.train(X_train, Y_train, opts.batch_size, lr)

    acc = np.mean(np.argmax(Y_valid, axis=1) == dnn.predict(X_valid) )
    acc_all.append(acc)

    print "Epoch %d, lr = %.4f, accuracy = %f" %(i+1, lr, acc)

    # dump intermediate model and log per 100 epoch
    if( np.mod( (i+1), 100) == 0):
        model_filename = os.path.join(opts.model_dir, "epoch%d.model"%(i+1))
        dnn_save_model(model_filename, dnn)
        
        log_filename = '../log/%s.log' %parameters
        print "Save %s" %log_filename
        np.savetxt(log_filename, acc_all, fmt='%.7f')
    
    lr *= opts.lr_decay

te = time.time()
report_time(ts, te)


# save model
model_filename = os.path.join(opts.model_dir, "epoch%d.model"%opts.epoch)
dnn_save_model(model_filename, dnn)

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
test_filename = '../feature/test.%s' %opts.feature
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s.csv' %parameters

Y_pred = dnn.predict(X_test)
dnn_save_label('../frame/test.frame', output_filename, Y_pred, opts.label_type)

