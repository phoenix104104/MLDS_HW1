from util import dnn_load_data, dnn_save_data
import numpy as np

class FRAME:
    def __init__(self):
        self.input  = ""    # complete input string
        self.ss     = ""    # speaker_sentence
        self.id     = -1    # frame id
        self.index  = -1 # index in input list


def make_frame(frame_str, index):
    frame = FRAME()
    fs = frame_str.split("_")
    frame.input = frame_str
    frame.ss    = fs[0] + "_" + fs[1]
    frame.id    = int(fs[2])
    frame.index = index
    return frame


fm = 4  # number of frame to merge


feature_name = 'fbank'
feature_filename = '../feature/train.%s' %feature_name
X = dnn_load_data(feature_filename) 

(N, D) = X.shape
x_zero = np.zeros(X[0].shape)
y_zero = 'zero'

frame_filename = '../frame/train.frame'
print "Load %s" %frame_filename
input_list = np.loadtxt(frame_filename, dtype='str')

N = len(input_list)
frame_list = []
for i in range(N):
    frame = make_frame(input_list[i], i)
    frame_list.append(frame)
    
X_new = np.zeros( (N, D*(2*fm+1)) )
for i in range(N):
    frame_curr = frame_list[i]
    x_list = []
    for j in range(i-fm, i+fm+1):
        if(j < 0 or j >= N):
            x_list.append(x_zero)
        else:
                
            frame_candidate = frame_list[j]
            if(frame_curr.ss != frame_candidate.ss):
                x_list.append(x_zero)  
            else:
                x_list.append(X[frame_candidate.index])
    
    X_new[i] = np.concatenate(x_list)


output_filename = '../feature/train.%s%d' %(feature_name, fm)
dnn_save_data(output_filename, X_new)

