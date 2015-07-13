from util import smooth_label, load_csv, save_csv, OPTS


data_dir = '../pred'

n_frame_to_smooth = 2

epoch = 1000      # use [model_dir]/epoch.model 

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

opts.hidden = [1024, 1024]

opts.data_size = '1M'
opts.feature = 'fbank4.norm'
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


filename = '../pred/%s_epoch%d.old.csv' %(parameters, epoch)
data = load_csv(filename)

smooth_y = smooth_label(data, n_frame_to_smooth)
for i in range(len(data)):
    data[i][1] = smooth_y[i]

filename = '../pred/%s_epoch%d.smooth%d.old.csv' %(parameters, epoch, n_frame_to_smooth)
header = ["Id", "Prediction"]
save_csv(filename, header, data)
