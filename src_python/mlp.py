"""
 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


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
        #self.output = ((
        #    lin_output if activation is None
        #    else activation(lin_output)),(
        #    lin_output_NN if activation is None
        #            else activation(lin_output_NN))
        #)
        
        self.output = (
            (lin_output,lin_output_NN) if activation is None
            else (activation(lin_output),activation(lin_output_NN))
        )

        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

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
                layer_input = self.sigmoid_layers[-1].output
            
            
            sigmoid_layer = HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )
            
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out
        )
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

        
        # end-snippet-3


def save_parameters(params,save_file):
    cPickle.dump(params, save_file, -1)  


def load_dataset(i):
    path1 = "/home/yannis/Desktop/DNN_py/data" + repr(i) + ".h5"  
    path2 = "/home/yannis/Desktop/DNN_py/labels" + repr(i) + ".h5"
    path3 = "/home/yannis/Desktop/DNN_py/neighbors" + repr(i) + ".h5"
    path4 = "/home/yannis/Desktop/DNN_py/wij_diff" + repr(i) + ".h5"

    print '... loading dataset ' + repr(i)

    data = numpy.array(h5py.File(path1,"r")["data"])
    labels = numpy.array(h5py.File(path2,"r")["labels"])
    neighbors = numpy.array(h5py.File(path3,"r")["data"])
    wij = numpy.array(h5py.File(path4,"r")["Wij_diff"])

    labels = labels - 1
    labels = labels.T.astype(int)
    
    train_set = (data,labels)
    train_set_x, train_set_y = shared_dataset(train_set)
    train_set_x_NN, train_set_y_NN = shared_dataset((neighbors,None))
    wij_sh = theano.shared(numpy.cast[theano.config.floatX](wij),name='wij_sh',borrow=True)

    return (train_set_x,train_set_y,train_set_x_NN,wij_sh)
    

def test_mlp(learning_rate_init=0.05, L1_reg=0.00, L2_reg=0.0001, n_epochs=2,
             batch_size=40, hidden_layers_sizes=[1024, 1024, 1024, 1024, 40], momentum=0.6, gamma=1.0, path="/home/yannis/Desktop/dnn_params", cost_path="/home/yannis/Desktop/obj_function",num_datasets=6):
    """
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    
   """
    save_file = open(path,'wb')
    save_cost = open(cost_path,'wb')
    

    dataset_idx = 1  #dataset index
    
    (train_set_x,train_set_y,train_set_x_NN,wij_sh) = load_dataset(dataset_idx)

    valid_test_data = numpy.array(h5py.File("/home/users/gchalk/DNN_py/dataset/test_data.h5","r")["data"])
    valid_test_labels = numpy.array(h5py.File("/home/users/gchalk/DNN_py/dataset/test_labels_1D.h5","r")["labels"])
 
    valid_test_labels = valid_test_labels - 1
    valid_test_labels = valid_test_labels.T.astype(int)
    
    test_set = (valid_test_data[1:300000,:],valid_test_labels[1:300000])
    valid_set = (valid_test_data[300001:600000,:],valid_test_labels[300001:600000])
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    gamma_sh = theano.shared(numpy.cast[theano.config.floatX](gamma), name='gamma_sh', borrow=True)
    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    batch_size_NN = wij_sh.shape[1]*batch_size 

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    x_NN = T.matrix('x_NN')
    l_rate = T.scalar('lr',dtype=theano.config.floatX)
    g = T.scalar('g',dtype=theano.config.floatX)
    wij_diff = T.matrix('wij_diff')

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=(x,x_NN),
        n_in=39,
        hidden_layers_sizes=hidden_layers_sizes,
        n_out=214
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y, g, wij_diff)
        #+ L1_reg * classifier.L1
        #+ L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: (test_set_y.flatten())[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: (valid_set_y.flatten())[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    #updates = [
    #    (param, param - learning_rate * gparam)
    #    for param, gparam in zip(classifier.params, gparams)
    #]

    updates = []
    for param,gparam in zip(classifier.params,gparams):
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - l_rate*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*gparam))

                     
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, theano.Param(l_rate,default=learning_rate_init)],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: (train_set_y.flatten())[index * batch_size: (index + 1) * batch_size],
            x_NN: train_set_x_NN[index * batch_size_NN: (index + 1) * batch_size_NN],
            g: gamma_sh,            
            wij_diff: wij_sh[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 9000000  # look as this many examples regardless 
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    cost_evolution = []
    epoch = 0
    done_looping = False
    n_epochs_init = n_epochs    

    while (epoch < n_epochs) and (not done_looping):

        if (epoch % 2 == 0):
            save_parameters(classifier.params,save_file)

        learning_rate = learning_rate_init*0.9995*math.exp(-epoch)
        epoch = epoch + 1
	
	for minibatch_index in xrange(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index,learning_rate)
            
            cost_evolution.append(minibatch_avg_cost)
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # validate at the end of each epoch
            if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, learning rate %f, validation error %f %%' %
                    (
                     epoch,
                     minibatch_index + 1,
                     n_train_batches,
                     learning_rate,
                     this_validation_loss * 100.
                    ) 
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                              in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                          'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                          test_score * 100.))
                    
            	
                        
                      


	if patience <= iter:
                done_looping = True
                save_parameters(classifier.params,save_file)
                break

        if (epoch == n_epochs):
	    if (dataset_idx != num_datasets):
            	dataset_idx = dataset_idx + 1
	    else:
		break
            n_epochs += n_epochs_init # equivalent to epoch = 0
            save_parameters(classifier.params,save_file) 
            (train_set_x,train_set_y,train_set_x_NN,wij_sh) = load_dataset(dataset_idx)                     

    end_time = timeit.default_timer()

    save_parameters(classifier.params,save_file)            
    save_file.close()            
    save_parameters(cost_evolution,save_cost)
    save_cost.close()

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
