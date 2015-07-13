import numpy as np
from util import dnn_load_data, dnn_save_data
from sklearn.preprocessing import StandardScaler
import os, sys



if __name__ == "__main__":
    

    #data_dir = '../feature'
    data_dir = '../../hw2/hw1_feature'
    feature = 'fbank7.norm.nn2048_2048_48.L3'
    filename = os.path.join(data_dir, 'train.%s'%feature)
    X = dnn_load_data(filename)

    print "Standard Normalization..."
    scaler = StandardScaler().fit(X)
    
    X = scaler.transform(X)
    
    filename = os.path.join(data_dir, 'train.%s.norm'%feature)
    dnn_save_data(filename, X)

    for t in ["test", 'test.old']:
        filename = os.path.join(data_dir, '%s.%s'%(t, feature) )
        X = dnn_load_data(filename)
    
        X = scaler.transform(X)

        filename = os.path.join(data_dir, '%s.%s.norm'%(t, feature) )
        dnn_save_data(filename, X)


