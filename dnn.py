import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import sys, os
from pickle import dump, load

srng = MRG_RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_init(shape):     
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def random_init_bias(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01),broadcastable=[True, False])

def zero_init(shape):
    return theano.shared(floatX(np.zeros(shape)))

def zero_init_bias(shape):
    return theano.shared(floatX(np.zeros(shape)),broadcastable=[True, False])

def init_from_pretrain(W):
    return theano.shared(floatX(W))

def sigmoid(X):
    return (T.exp(X)/(T.exp(X)+1))

def tanh(X):
    return T.tanh(X)

def ReLU(X):
    return T.maximum(X, 0.0)

def softmax():
    pass

def cross_entropy():
    # TODO: need a cost function, can be something other than cross_entropy
    pass

def cost_func(X,Y):
    # merging softmax and cross_entropy to avoid dimension mismatch
    x_exp = T.exp(X)
    return T.mean(T.log(T.sum(x_exp, axis=1)) - T.sum(X*Y, axis=1))

def sgd(cost, W, lr, opts):

    w_reg   = 1 - opts.weight_decay * lr;
    mu      = opts.momentum
     
    grads = T.grad(cost=cost, wrt=W)
    updates = []
    for w, g in zip(W, grads):
        w_m = theano.shared(w.get_value()*0.0, broadcastable=w.broadcastable) # initialize momentum
        #updates.append([w, w_reg * w - g * lr]) # original sgd
        #updates.append([w_m, mu * w_m - lr * g])
        #updates.append([w, w_reg * w + w_m])
        updates.append([w, w_reg * w - lr * w_m])
        updates.append([w_m, mu * w_m + (1 - mu) * g])

    return updates

def RMSProp(cost, W, lr, opts):
    
    eps = 1e-6
    w_reg = 1 - opts.weight_decay * lr;
    alpha = opts.rmsprop_alpha

    grads = T.grad(cost=cost, wrt=W)
    updates = []
    for w, g in zip(W, grads):
        w_acc = theano.shared(w.get_value()*0.0, broadcastable=w.broadcastable)
        w_acc_next = alpha * (w_acc) + (1 - alpha) * (g ** 2)
        g_scaling = T.sqrt(w_acc_next + eps)
        g = g / g_scaling
        updates.append([w_acc, w_acc_next])
        updates.append([w, w_reg * w - lr * g])

    return updates


def dropout(X, drop_prob=0.0, is_training=1):

    if( drop_prob > 0.0):
        retain_prob = 1 - drop_prob
        if( is_training ):
            mask = srng.binomial(n=1, size=X.shape, p=drop_prob)
            X *= T.cast(mask, theano.config.floatX)
            #X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            #X /= retain_prob
        else:
            X *= retain_prob

    return X

def model(X, W, B, act, dropout_prob, is_training):
    
    layer_num = len(W)
    H = [[]] * layer_num

    hidden = dropout(X, dropout_prob, is_training)
    for i in range(layer_num-1):
        hidden = act(T.dot(hidden, W[i]) + B[i])
        hidden = dropout(hidden, dropout_prob, is_training)
        H[i] = hidden
       
    # last layer
    H[layer_num-1] = T.dot(hidden, W[layer_num-1]) + B[layer_num-1]

    return H
    

class DNN:
    def __init__(self, opts):

        self.X = T.matrix(dtype=theano.config.floatX)
        self.Y = T.matrix(dtype=theano.config.floatX)

        self.W  = []
        self.B  = []
        #self.mW = []
        #self.mB = []
        for i in range(len(opts.structure)-1):
            if( opts.pretrain ):
                w = init_from_pretrain(opts.W_init[i])
            else:
                w = random_init((opts.structure[i], opts.structure[i+1]))
            
            self.W.append(w)
            #self.mW.append(zero_init(w.shape))

        for i in range(1,len(opts.structure)):
            b = random_init_bias((1, opts.structure[i]))
            self.B.append(b)
            #self.mB.append(zero_init_bias(b))
        
        self.opts = opts
        self.lr = theano.shared(np.cast[theano.config.floatX](opts.learning_rate))

        if(opts.activation == 'sigmoid'):
            self.act = sigmoid
        elif(opts.activation == 'tanh'):
            self.act = tanh
        elif(opts.activation == 'ReLU'):
            self.act = ReLU
        else:
            print "Unknown activation %s" %opts.act
            sys.exit(1)


        # training model
        self.H_train = model(self.X, self.W, self.B, self.act, self.opts.dropout_prob, 1)
        self.H_test  = model(self.X, self.W, self.B, self.act, self.opts.dropout_prob, 0)
        
        self.hidden = []
        for h in self.H_test:
            self.hidden.append(theano.function(inputs=[self.X], outputs=h, allow_input_downcast=True))


        self.cost = cost_func(self.H_train[-1], self.Y)

        if(opts.update_grad == 'sgd'):
            self.updates = sgd(self.cost, self.W+self.B, self.lr, self.opts)
        elif(opts.update_grad == 'rmsprop'):
            self.updates = RMSProp(self.cost, self.W+self.B, self.lr, self.opts)
        else:
            print "Unknown gradient update method %s" %opts.update_grad
            sys.exit(1)

        self.y_pred = T.argmax(self.H_test[-1], axis=1)

        self.train_batch = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
    
        self.predict_batch = theano.function(inputs=[self.X], outputs=self.y_pred, allow_input_downcast=True)
        
        print "================================================"
        print "NN structure \t\t= %s" %("-".join(str(s) for s in opts.structure))
        print "Batch size \t\t= %s" %(str(opts.batch_size))
        print "Initial learning rate \t= %s" %(str(opts.learning_rate))
        print "Momentum \t\t= %s" %(opts.momentum)
        print "Weight decay \t\t= %s" %(opts.weight_decay)
        print "RMSProp alpha \t\t= %s" %(opts.rmsprop_alpha)
        print "Dropout probability \t= %s" %(str(opts.dropout_prob))
        print "Activation function \t= %s" %(opts.activation)
        print "Gradient update \t= %s" %(opts.update_grad)
        print "RBM pretraining \t= %s" %(opts.pretrain)
        print "================================================"

    def train(self, X_train, Y_train, batch_size, learning_rate):
        
        self.set_learning_rate(learning_rate)        
        starts = range(0, len(X_train), batch_size)
        ends = range(batch_size, len(X_train), batch_size) + [len(X_train)]
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

    def predict(self, X, batch_size=1000):
        N = len(X)
        starts = range(0, N, batch_size)
        ends   = range(batch_size, N, batch_size) + [N]
        
        pred_list = []
        for st, ed in zip(starts, ends):
            x = X[st:ed]
            pred_list.append( self.predict_batch(x) )

        pred = np.concatenate(pred_list)
        return pred

    def get_hidden_feature(self, X, l, batch_size=1000):
        N = len(X)
        starts = range(0, N, batch_size)
        ends   = range(batch_size, N, batch_size) + [N]
        
        output = []
        for st, ed in zip(starts, ends):
            x = X[st:ed]
            output.append( self.hidden[l-1](x) )

        output = np.concatenate(output)
        
        return output

    
