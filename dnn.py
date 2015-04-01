import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import sys, os
from pickle import dump, load

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_init(shape):     
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def random_init_bias(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01),broadcastable=[True, False])

def sigmoid(X):
    # TODO
    return (T.exp(X)/(T.exp(X)+1))

def tanh(X):
    return T.tanh(X)

def softmax():
    # TODO
    pass

def cross_entropy():
    # TODO: need a cost function, can be something other than cross_entropy
    pass

def cost_func(X,Y):
    # merging softmax and cross_entropy to avoid dimension mismatch
    x_exp = T.exp(X)
    return T.mean(T.log(T.sum(x_exp, axis=1)) - T.sum(X*Y, axis=1))

def sgd(cost, params, lr):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def rectify(X):
    return T.maximum(X, 0.0)

def dropout(X, drop_prob=0.0):
    if( drop_prob > 0.0):
        retain_prob = 1.0 - drop_prob
        #X *= np.random.binomial(1, retain_prob, X.shape)
        X *= MRG_RandomStreams().binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob

    return X

class DNN:
    def __init__(self, structure, learning_rate=0.05, batch_size=100, act='sigmoid', dropout_prob=[0.0, 0.0], model_dir='../model'):
        self.X = T.matrix(dtype=theano.config.floatX)
        self.Y = T.matrix(dtype=theano.config.floatX)

        self.W = []
        self.B = []
        for i in range(len(structure)-1):
            w = random_init((structure[i], structure[i+1]))
            self.W.append(w)
        for i in range(1,len(structure)):
            b = random_init_bias((1,structure[i]))
            self.B.append(b)

        self.structure = structure
        self.lr = theano.shared(np.cast[theano.config.floatX](learning_rate))
        self.batch_size = batch_size
        self.input_drop_prob  = dropout_prob[0]
        self.hidden_drop_prob = dropout_prob[1]
        self.model_dir = model_dir

        if(act == 'sigmoid'):
            self.act = sigmoid
        elif(act == 'tanh'):
            self.act = tanh
        elif(act == 'ReLU'):
            self.act = rectify
        else:
            print "Unknown activation %s" %act
            sys.exit(1)

        # training model
        hidden = dropout(self.X, self.input_drop_prob)
        layer_num = len(self.W)
        for i in range(layer_num-1):
            hidden = self.act(T.dot(hidden, self.W[i]) + self.B[i])
            hidden = dropout(hidden, self.hidden_drop_prob)

        self.model = T.dot(hidden, self.W[layer_num-1]) + self.B[layer_num-1]
        
        self.cost = cost_func(self.model, self.Y)

        self.updates = sgd(self.cost, self.W+self.B, self.lr)

        self.y_pred = T.argmax(self.model, axis=1)

        self.train_batch = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
    
        self.predict = theano.function(inputs=[self.X], outputs=self.y_pred, allow_input_downcast=True)
        
        print "NN structure: %s" %("-".join(str(s) for s in structure))
        print "Activation function = %s" %act
        print "Dropout probability = (%s, %s)" %(str(dropout_prob[0]), str(dropout_prob[1]))

    def train(self, X_train, Y_train, learning_rate):
        
        self.set_learning_rate(learning_rate)        
        starts = range(0, len(X_train), self.batch_size)
        ends = range(self.batch_size, len(X_train), self.batch_size) + [len(X_train)]
        randperm = np.random.permutation(len(X_train))

        for start, end in zip(starts, ends): 
            X_round = []
            Y_round = []
            for rpidx in randperm[start:end]:
                X_round.append(X_train[rpidx])
                Y_round.append(Y_train[rpidx])

            self.train_batch(X_round,Y_round)


    def set_learning_rate(self, learning_rate):
        self.lr.set_value(learning_rate)

    def batch_predict(self, X, block_size=1000):
        N = len(X)
        starts = range(0, N, block_size)
        ends   = range(block_size, N, block_size) + [N]
        
        pred_list = []
        for st, ed in zip(starts, ends):
            x = X[st:ed]
            pred_list.append( self.predict(x) )

        pred = np.concatenate(pred_list)
        return pred

    
