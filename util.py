import numpy as np
import mapping
import csv
from pickle import dump, load

sil = 'L'

def mode(A):
    F = dict()
    for a in A:
        if F.has_key(a):
            F[a] += 1
        else:
            F[a] = 1

    max_key = ''
    max_count = 0
    for a in F.keys():
        if( F[a] > max_count ):
            max_count = F[a]
            max_key = a

    return max_key

 
class UTTERANCE:
    def __init__(self):
        self.name           = ""    # audio filename
        self.id_list        = []    # frame id
        self.phone_list     = []    # frame phone
        self.phone_sequence = ''    # merged phone

    def trimming(self):
        phone_sequence = ""
        phone_curr = ""
        # trimming
        for phone in self.phone_list:
            if phone != phone_curr:
                phone_curr = phone
                phone_sequence += phone
        
        # eliminate sil at the beginning and the end
        st = 0
        ed = len(phone_sequence)
        if( phone_sequence[0] == sil ):
            st = 1
        if( phone_sequence[-1] == sil ):
            ed -= 1
        
        self.phone_sequence = phone_sequence[st:ed]

    def smoothing(self, n_frame):
        N = len(self.phone_list)
        new_phone_list = [id for id in self.phone_list]

        for i in range(n_frame, N-n_frame) :
            collect = self.phone_list[i-n_frame:i+n_frame+1]
            new_phone_list[i] = mode(collect)
        
        self.phone_list = new_phone_list

def extract_audio_name(string):
    s = string.split('_')
    person = s[0]
    audio = s[1]
    return person, audio

def save_csv(output_filename, header, data):

    with open(output_filename, 'w+') as file:
        print "Save %s" %output_filename
        writer = csv.writer(file)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

def load_csv(input_filename):
    
    with open(input_filename, 'r') as f:
        print "Load %s" %input_filename
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)

    header = data[0]
    data = data[1:]
    
    return data


def smooth_label(data, n_frame):
    
    audio_all = []
    name_current = ""
    for name, y in data:
        person_name, audio_name = extract_audio_name(name)
        name = '%s_%s' %(person_name, audio_name)
        if( name != name_current ):
            name_current = name
            audio = UTTERANCE()
            audio.name = name
            audio_all.append(audio)

        audio.phone_list.append(y)

    output = []
    for audio in audio_all:    
        audio.smoothing(n_frame)
        output += audio.phone_list
    
    return output 

class OPTS:
    def __init__(self):
        self.epoch = 0

def report_time(ts, te):
    D = 24 * 60 * 60
    H = 60 * 60
    M = 60

    total_sec = te - ts;

    day  = np.floor(total_sec / D)
    hour = np.floor( np.mod(total_sec, D) / H )
    min  = np.floor( np.mod(np.mod(total_sec, D), H) / M )
    sec  = np.mod( np.mod(np.mod(total_sec, D), H), M)
    print "Elapsed time is %d Days, %d Hours, %d Mins, %d secs" %(day, hour, min, sec)

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

def dnn_save_data(output_filename, X):
    print "Save %s" %output_filename
    np.savetxt(output_filename, X, fmt='%.7f') 

def dnn_save_model(model_filename, model):

    with open(model_filename, 'w+') as f:
        print "Save %s" %model_filename
        dump(model, f)

def dnn_load_model(model_filename):

    with open(model_filename, 'r') as f:
        print "Load %s" %model_filename
        model = load(f)
        return model

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
    
    with open(output_filename, 'w+') as file:
        print "Save %s" %output_filename
        writer = csv.writer(file)
        writer.writerow(["Id", "Prediction"])
        for row in csv_data:
            writer.writerow(row)

    return predict_label
    

def dnn_save_feature(output_filename, feature, frame):
    
    N = len(frame)
    with open(output_filename, 'w') as f:
        print "Save %s" %output_filename
        for i in range(N):
            f.write("%s " %frame[i])
            f.write(" ".join("%.8f" %x for x in feature[i]))
            f.write("\n")



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
