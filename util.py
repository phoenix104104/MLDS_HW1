import numpy as np
import mapping
import csv

def one_hot(y, n_class):

#   map label vector from [1; 3; 10; ...]
#   to  [ [1, 0, 0, ..., 0];
#         [0, 0, 1, ..., 0];
#         [0, 0, 0, ..., 1];
#         ...               ];

    if type(y) == list:
        y = np.array(y)

    y = y.flatten()
    o_h = np.zeros( (len(y), n_class) )
    o_h[np.arange(len(y)), y] = 1

    return o_h

def dnn_load_data(train_filename, label_filename="", n_class=""):

#   Usage: 
#       1. Load training data (with label):
#
#           (X, y) = dnn_load_data(train_filename, label_filename, n_class)
#
#       2. Load testing data (without label):
#
#           X = dnn_load_data(test_filename)


    print "Load %s" %train_filename
    X = np.loadtxt(train_filename, dtype='float')

    if( label_filename != "" ):

        print "load %s" %label_filename
        Y = np.loadtxt(label_filename, dtype='int')
        Y = one_hot(Y, n_class)

        return (X, Y)

    else:
        return (X)

def dnn_save_label(frame_filename, output_filename, predict_index, label_type):
    if( label_type == '39' ):
        predict_label = [str(x) for x in predict_index]

    elif( label_type == '48' ):
        predict_label = map_index_to_48(predict_index)
        predict_label = downcast_48_to_39(predict_label)

    elif( label_type == 'state'):
        predict_label = map_index_to_state(predict_index)
        predict_label = downcast_state_to_39(predict_label)
    
    else:
        sys.exit("Unknown label type %s" %label_type)
    
    # save csv file
    frame_list = np.loadtxt(frame_filename, dtype='str')
    csv_data = []
    for i in range(len(predict_label)):
        csv_data.append([frame_list[i], predict_label[i]])
    
    with open(output_filename, 'w') as file:
        print "Save %s" %output_filename
        writer = csv.writer(file)
        writer.writerow(["Id", "Prediction"])
        for row in csv_data:
            writer.writerow(row)

    return predict_label
    

def downcast_48_to_39(label_list):
    D = mapping.dict_48_39
    for i in range(len(label_list)):
        label_list[i] = D[label_list[i]]

    return label_list


def downcast_state_to_39(label_list):
    D = mapping.dict_state_39
    for i in range(len(label_list)):
        label_list[i] = D[label_list[i]]

    return label_list


def map_39_to_index(label_list):
    D = mapping.dict_39_index
    
    n = len(label_list)
    index_list = np.ndarray(shape=(n, 1), dtype='int')
    for i in range(n):
        index_list[i] = D[label_list[i]]
    
    return index_list


def map_48_to_index(label_list):
    D = mapping.dict_48_index

    n = len(label_list)
    index_list = np.ndarray(shape=(n, 1), dtype='int')
    for i in range(n):
        index_list[i] = D[label_list[i]]
    
    return index_list


def map_state_to_index(label_list):
    n = len(label_list)
    index_list = np.ndarray(shape=(n, 1), dtype='int')
    for i in range(n):
        index_list[i] = int(label_list[i])
    
    return index_list


def map_index_to_39(index_list):
    D = mapping.dict_index_39
    
    n = len(index_list)
    label = []
    for i in range(n):
        label.append(D[index_list[i]])
    
    return label


def map_index_to_48(index_list):
    D = mapping.dict_index_48
    
    n = len(index_list)
    label = []
    for i in range(n):
        label.append(D[index_list[i]])
    
    return label

def map_index_to_state(index_list):
    
    n = len(index_list)
    label = []
    for i in range(n):
        label.append(str(index_list[i]))
    
    return label
