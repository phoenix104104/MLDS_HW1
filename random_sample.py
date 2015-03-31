import numpy as np
import random

train_num = 1000000
valid_num = 124823

N = 1124823
sample_index = random.sample(range(N), train_num + valid_num)
train_index = sample_index[:train_num]
valid_index = sample_index[train_num+1:]

filename = 'train1M.list'
print "Save %s" %filename
np.savetxt(filename, train_index, fmt='%d')

filename = 'valid1M.list'
print "Save %s" %filename
np.savetxt(filename, valid_index, fmt='%d')

