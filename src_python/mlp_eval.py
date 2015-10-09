import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import math
import h5py
import cPickle

from logistic_sgd import LogisticRegression, shared_dataset


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input_x,self.input_xNN = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input[0], self.W) + self.b
        lin_output_NN = T.dot(input[1], self.W) + self.b
        
        self.output = (
            (lin_output,lin_output_NN) if activation is None
            else (activation(lin_output),activation(lin_output_NN))
        )

        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):

    def __init__(self, rng, input, n_in, hidden_layers_sizes, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.params = []
        self.sigmoid_layers = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        ws = []
        bs = []
        save_file1 = open('/home/users/gchalk/DNN_py/params_cpu')
        for i in xrange(9):
            a = cPickle.load(save_file1)
            ws.append(a)
            b = cPickle.load(save_file1)
            bs.append(b)
        save_file1.close()


        for i in xrange(self.n_layers):
            #input layer
            if i == 0: 
                input_size = n_in
            #hidden layer
            else:
                input_size = hidden_layers_sizes[i - 1]
            
            if i == 0:
                layer_input = input
            else:
                layer_input = self.sigmoid_layers[i-1].output


            sigmoid_layer = HiddenLayer(
                 rng=rng,
                 input=layer_input,
                 n_in=input_size,
                 n_out=hidden_layers_sizes[i],
                 W=ws[i],
                 b=bs[i],
                 activation=T.nnet.sigmoid
            )

            print '... loaded W, b'


            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
#######
        #W=None
        #b=None
        self.logRegressionLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out,
            W=ws[-1],
            b=bs[-1]
        )
	
#######
        self.params.extend(self.logRegressionLayer.params)

        # end-snippet-2 start-snippet-3
       
        for i in xrange(self.n_layers):
            # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
            self.L1 = abs(self.sigmoid_layers[i].W).sum()
            # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
            self.L2_sqr = (self.sigmoid_layers[i].W ** 2).sum()
           
        
        self.L1 = (self.L1 + abs(self.logRegressionLayer.W).sum())
        self.L2_sqr = (self.L2_sqr + (self.logRegressionLayer.W ** 2).sum())

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = ( self.logRegressionLayer.errors
        )
#######
        self.predictions = ( self.logRegressionLayer.predictions
        )
#######
        # end-snippet-3


def save_parameters(params,save_file):
    cPickle.dump(params, save_file, -1)  


def load_dataset(i):
    path1 = "/home/yannis/Desktop/data" + repr(i) + ".h5"  
    path2 = "/home/yannis/Desktop/labels" + repr(i) + ".h5"
    path3 = "/home/yannis/Desktop/neighbors" + repr(i) + ".h5"
    path4 = "/home/yannis/Desktop/wij_diff" + repr(i) + ".h5"

    print '... loading dataset ' + repr(i)

    data = numpy.array(h5py.File(path1,"r")["data"])
    labels = numpy.array(h5py.File(path2,"r")["labels"])
    neighbors = numpy.array(h5py.File(path3,"r")["data"])
    wij = numpy.array(h5py.File(path4,"r")["Wij_diff"])

    #Afterwards add 1 for the correct label
    labels = labels - 1
    labels = labels.T.astype(int)
    
    train_set = (data,labels)
    train_set_x, train_set_y = shared_dataset(train_set)
    train_set_x_NN, train_set_y_NN = shared_dataset((neighbors,None))
    wij_sh = theano.shared(numpy.cast[theano.config.floatX](wij),name='wij_sh',borrow=True)

    return (train_set_x,train_set_y,train_set_x_NN,wij_sh)
    




def eval_mlp(batch_size=4,hidden_layers_sizes=[1024,1024,1024,1024,1024,1024,2048,2048]):
    
    eval_data = numpy.array(h5py.File("/home/users/gchalk/DNN_py/eval_92norm.h5","r")["mfcc"])
    eval_set = (eval_data.transpose(),None)
   
    #priors = numpy.array(h5py.File("/home/users/gchalk/DNN_py/priors134.h5","r")["priors"])

    #print priors.shape
    
    eval_set_x, valid_set_y = shared_dataset(eval_set)
    
       
    #print '... building the model'

    # allocate symbolic variables for the data
    x = T.matrix('x')  
    x_NN = T.matrix('x_NN')
    
    rng = numpy.random.RandomState(1234)

    numpy.set_printoptions(threshold='nan')
    numpy.set_printoptions(precision=10)
    numpy.set_printoptions(suppress=True)
    numpy.set_printoptions(linewidth=150000)  ##set according to output dims

    classifier = MLP(
           rng=rng,
           input=(x,x_NN),
           n_in=39,
           hidden_layers_sizes=hidden_layers_sizes,
           n_out=132
    )


    n_eval_batches = eval_set_x.get_value(borrow=True).shape[0] / batch_size
    index = T.lscalar()


    evaluate_model = theano.function(
        inputs=[index],
        outputs=classifier.predictions(),
        givens={
            x: eval_set_x[index * batch_size: (index + 1) * batch_size],
            x_NN: eval_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '... evaluating model'
    save_file = open("/home/users/gchalk/predictions.txt", "w")

    for minibatch_index in xrange(n_eval_batches):
        a = evaluate_model(minibatch_index)
        for i in xrange(len(a)):
           save_file.write(str(a[i,:])+"\n")
#           save_file.write(str(numpy.divide(a[i,:],priors))+"\n")

    save_file.close()
    
if __name__ == '__main__':
    eval_mlp()
 
