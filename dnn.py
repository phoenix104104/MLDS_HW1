import theano
from theano import tensor as T
import numpy as np

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_init(shape):     
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def random_init_bias(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01),broadcastable=[True, False])

def sigmoid():
    # TODO

def softmax():
    # TODO

def cross_entropy():
    # TODO: need a cost function, can be something other than cross_entropy

class DNN:
    def __init__(self, structure, learning_rate=0.05, epoch=10, batch_size=100):
        self.X = T.fmatrix()
        self.Y = T.fmatrix()

        self.W = []
        self.B = []
        for i in range(len(structure)-1):
            w = random_init((structure[i], structure[i+1]))
            self.W.append(w)
        for i in range(1,len(structure)):
            b = random_init_bias((1,structure[i]))
            self.B.append(b)

        self.structure = structure
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        
        # TODO: model
        temp = self.X
        layer_num = len(self.W)
        for i in range(layer_num-1):
            temp = sigmoid(T.dot(temp, self.W[i]) + self.B[i])
        self.model = softmax(T.dot(temp, self.W[layer_num-1]) + self.B[layer_num-1])

        # TODO: updates(sgd)
        
        # TODO: y_pred
        self.y_pred = T.argmax(self.model, axis=1)

        # TODO: cost

        self.train_epoch = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
    
        self.predict = theano.function(inputs=[self.X], outputs=self.y_pred, allow_input_downcast=True)


    def train(self, X_train, Y_train, X_valid, Y_valid):
        for i in range(self.epoch):
            # TODO: batch
            starts = range(0, len(X_train), self.batch_size)
            ends = range(self.batch_size, len(X_train), self.batch_size) + [len(X_train)]
            for start, end in zip(starts, ends): 
                self.train_epoch(X_train[start:end],Y_train[start:end])
        acc = np.mean(np.argmax(Y_valid, axis=1) == self.predict(X_valid) )   
        
        print "Epoch %d, accuracy = %f" %(i, acc) 


