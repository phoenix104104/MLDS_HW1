import theano
from theano import tensor as T
import numpy as np

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_init(shape):     
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sigmoid():
    # TODO

def softmax():
    # TODO

class DNN:
    def __init__(self, structure, learning_rate=0.05, epoch=10, batch_size=100):
        self.X = T.fmatrix()
        self.Y = T.fmatrix()

        self.W = []
        self.B = []
        for i in range(len(structure)-1):
            w = random_init((structure[i], structure[i+1]))
            self.W.append(w)
        
        self.structure = structure
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        
        # TODO: model
        
        # TODO: updates(sgd)

        # TODO: y_pred
        
        # TODO: cost

        self.train_epoch = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
    
        self.predict = theano.function(inputs=[self.X], outputs=self.y_pred, allow_input_downcast=True)


    def train(self, X_train, Y_train, X_valid, Y_valid):
        for i in range(self.epoch):
            # TODO: batch
        
        acc = np.mean(np.argmax(Y_valid, axis=1) == self.predict(X_valid) )   
        
        print "Epoch %d, accuracy = %f" %(i, acc) 


