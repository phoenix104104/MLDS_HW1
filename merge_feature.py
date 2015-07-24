from util import dnn_load_data
import numpy as np

for tv in ['train', 'valid']:

    X1 = dnn_load_data('../feature/%s1M.fbank'%tv)
    X2 = dnn_load_data('../feature/%s1M.mfcc'%tv)
    X = np.concatenate((X1, X2), axis=1)
    print "Save ../feature/%s1M.fm" %tv
    np.savetxt('../feature/%s1M.fm'%tv, X, fmt='%7f')

