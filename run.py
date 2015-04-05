import theano
from theano import tensor as T
import numpy as np
from util import dnn_load_data, dnn_save_label, report_time, dnn_save_model
from dnn import DNN
from pickle import dump, load
import time, sys, os

sys.setrecursionlimit(9999) # to dump large network
#---------- training script ----------#

epoch         = 1000
batch_size    = 100
learning_rate = 0.05
lr_decay      = 1
dropout_prob  = [0., 0.]
activation = 'sigmoid'
#activation = 'tanh'
#activation = 'ReLU'

hidden = [128, 128]

feature = 'fbank'
label_type = '48'
N_class = 48
data_size = '1M'

parameters = '%s_%s_%s_nn%s_%s_epoch%d_lr%s_decay%s_drop%s' \
              %(feature, label_type, data_size, \
                "_".join(str(h) for h in hidden), activation, \
                epoch, str(learning_rate), str(lr_decay), \
                "_".join(str(p) for p in dropout_prob) )

model_dir = '../model/%s_%s_%s_nn%s_%s_lr%s_decay%s_drop%s' \
              %(feature, label_type, data_size, "_".join(str(h) for h in hidden), activation, \
                str(learning_rate), str(lr_decay), "_".join(str(p) for p in dropout_prob) )

if( os.path.isdir(model_dir) ):
    print "Warning! Directory %s already exists!" %model_dir
    print ">>>>> ctrl+c to leave the program, or press any key to continue..."
    raw_input()
else:
    print "mkdir %s" %model_dir
    os.mkdir(model_dir)

if( data_size == 'all' ):
    train_filename = '../feature/train.%s' %(feature)
    train_labelname = '../label/train.%s.index' %(label_type)
else:
    train_filename = '../feature/train%s.%s' %(data_size, feature)
    train_labelname = '../label/train%s.%s.index' %(data_size, label_type)

X_train, Y_train = dnn_load_data(train_filename, train_labelname, N_class)

if( data_size == 'all' ):
    valid_filename = '../feature/valid1M.%s' %(feature)
    valid_labelname =  '../label/valid1M.%s.index' %(label_type)
else:
    valid_filename = '../feature/valid%s.%s' %(data_size, feature)
    valid_labelname =  '../label/valid%s.%s.index' %(data_size, label_type)

X_valid, Y_valid = dnn_load_data(valid_filename, valid_labelname, N_class)

(N_data, N_dim) = X_train.shape

structure = [N_dim] + hidden + [N_class]


print "Build DNN structure..."
dnn = DNN(structure, learning_rate, batch_size, activation, dropout_prob, model_dir)

# training
print "Start DNN training...(lr_decay = %s)" %(str(lr_decay))
ts = time.time()

acc_all = []
lr = learning_rate
for i in range(epoch):

    dnn.train(X_train, Y_train, lr)

    acc = np.mean(np.argmax(Y_valid, axis=1) == dnn.batch_predict(X_valid) )
    acc_all.append(acc)

    print "Epoch %d, lr = %.4f, accuracy = %f" %(i, lr, acc)

    # dump intermediate model and log per 100 epoch
    if( np.mod( (i+1), 100) == 0):
        model_filename = os.path.join(model_dir, "epoch%d.model"%(i+1))
        dnn_save_model(model_filename, dnn)
        
        log_filename = '../log/%s.log' %parameters
        print "Save %s" %log_filename
        np.savetxt(log_filename, acc_all, fmt='%.7f')
    
    lr *= lr_decay

te = time.time()
report_time(ts, te)


# save model
model_filename = os.path.join(model_dir, "epoch%d.model"%epoch)
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
test_filename = '../feature/test.fbank'
X_test = dnn_load_data(test_filename)

output_filename = '../pred/%s.csv' %parameters

Y_pred = dnn.batch_predict(X_test)
dnn_save_label('../frame/test.frame', output_filename, Y_pred, label_type)

