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

import numpy

import theano
import theano.tensor as T
import math
import h5py
import cPickle
import theano.printing




from theano.ifelse import ifelse
from collections import OrderedDict

from logistic_sgd_NN import LogisticRegression, shared_dataset

from io_func import smart_open
from model_io import _nnet2file, _file2nnet


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
        self.x , self.x_NN = input
        self.activation = activation
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

        lin_output = T.dot(self.x, self.W) + self.b
        if self.x_NN!=None:
            lin_output_NN = T.dot(self.x_NN, self.W) + self.b
        else:
            lin_output_NN = None

        if self.x_NN!=None:
            self.output = (
                (lin_output,lin_output_NN) if activation is None
                else (activation(lin_output),activation(lin_output_NN))
            )
        else:
            self.output = (
                (lin_output,None) if activation is None
                else (activation(lin_output),None)
            )

        # parameters of the model
        self.params = [self.W, self.b]


def _dropout_from_layer(rng, (x,x_NN), p, knn=None):

    """p is the probablity of dropping a unit
    """
    #srng = theano.tensor.shared_randomstreams.RandomStreams(
    #        rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = rng.binomial(n=1, p=1-p, size=x.shape)
    if (x_NN!=None and knn!=None):
        mask_NN = T.extra_ops.repeat(mask,knn,axis=0)   ####need to apply same mask over each sample + corresponding NN's, or should mask be (x.shape[1],1)->same for each sample in batch->same for all samples in x_NN
        output = (x * T.cast(mask, theano.config.floatX),
                  x_NN * T.cast(mask_NN, theano.config.floatX))
    else:
        output = (x * T.cast(mask, theano.config.floatX),None)

    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, dropout_rate, W=None, b=None,
                 activation=T.nnet.sigmoid, knn=None ):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.theano_rng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))

        self.output = _dropout_from_layer(self.theano_rng, self.output, p=dropout_rate, knn=knn) # conflict with self.output? vs self.output_inherited?


class MLP(object):
    """Multi-Layer Perceptron Class
    """

    def __init__(self, numpy_rng, input, n_in, hidden_layers_sizes, n_out, knn=None,  cfg=None, theano_rng=None, dropout_rates=None,activations=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type hidden_layers_sizes: vector of ints
        :param hidden_layers_sizes: number of hidden units in each hidden layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        #Build layers
        self.dropout_rates=dropout_rates
        self.cfg = cfg
        self.params = []
        self.x,self.x_NN = input
        self.y = T.ivector('y')
        self.hidden_layers_sizes = hidden_layers_sizes
        self.wij = T.fvector('wij')
        self.g = T.scalar('g',dtype=theano.config.floatX)
        self.knn = knn       
        if dropout_rates!=None:
            self.dropout_params = []
            self.dropout_layers = []
        self.L1 = theano.shared(numpy.cast[theano.config.floatX](0.0), name='L1', borrow=True)
        self.L2_sqr = theano.shared(numpy.cast[theano.config.floatX](0.0), name='L2_sqr', borrow=True)
        self.sigmoid_params = []
        self.sigmoid_layers = []

        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        if theano_rng==None:
            self.theano_rng = theano.tensor.shared_randomstreams.RandomStreams(numpy_rng.randint(2 ** 30))


        for i in xrange(self.n_layers):
            #input layer
            if i == 0: 
                input_size = n_in
            #hidden layer
            else:
                input_size = hidden_layers_sizes[i - 1]
            
            if i == 0:
               if dropout_rates!=None:
                    if self.x_NN!=None:
                        dropout_layer_input = _dropout_from_layer(self.theano_rng, input, p=dropout_rates[i], knn=self.knn)
                    else:
                        dropout_layer_input = _dropout_from_layer(self.theano_rng, input, p=dropout_rates[i])
               layer_input = input
            else:
               if dropout_rates!=None:
                    if self.x_NN!=None:
                       dropout_layer_input = (T.cast(1/(1-dropout_rates[i-1]), theano.config.floatX)*self.dropout_layers[i-1].output[0],T.cast(1/(1-dropout_rates[i-1]), theano.config.floatX)*self.dropout_layers[i-1].output[1])
                    else:
                       dropout_layer_input = (T.cast(1/(1-dropout_rates[i-1]), theano.config.floatX)*self.dropout_layers[i-1].output[0],None)
               layer_input = self.sigmoid_layers[i-1].output

            if activations!=None:
                 activation = activations[i]
            else:
                 activation = T.nnet.sigmoid


            # hidden layers for training network
            if dropout_rates!=None:
                if self.x_NN!=None:
                    dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                                       input=dropout_layer_input,
                                                       n_in=input_size,
                                                       n_out=hidden_layers_sizes[i],
                                                       activation=activation,
                                                       dropout_rate=dropout_rates[i],
                                                       knn=self.knn
                                                       )

                else:
                    dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                                       input=dropout_layer_input,
                                                       n_in=input_size,
                                                       n_out=hidden_layers_sizes[i],
                                                       activation=activation,
                                                       dropout_rate=dropout_rates[i]
                                                       )
                self.dropout_layers.append(dropout_layer)
                self.dropout_params.extend(dropout_layer.params)
                sigmoid_layer = HiddenLayer(
                    rng=numpy_rng,
                    input=layer_input,
                    n_in=input_size,
                    n_out=hidden_layers_sizes[i],
                    W=dropout_layer.W,
                    b=dropout_layer.b,
                    activation=activation
                )
            else:
                sigmoid_layer = HiddenLayer(
                    rng=numpy_rng,
                    input=layer_input,
                    n_in=input_size,
                    n_out=hidden_layers_sizes[i],
                    activation=activation
                )
            self.sigmoid_layers.append(sigmoid_layer)
            self.sigmoid_params.extend(sigmoid_layer.params)

        # The logistic regression layer gets as input the hidden units
        # of the last hidden layer

        if dropout_rates!=None:
            if self.x_NN!=None:
                input_logregr = (T.cast(1/(1-dropout_rates[i-1]), theano.config.floatX)*self.dropout_layers[-1].output[0],T.cast(1/(1-dropout_rates[i-1]), theano.config.floatX)*self.dropout_layers[-1].output[1])
            else:
                input_logregr = (T.cast(1/(1-dropout_rates[i-1]), theano.config.floatX)*self.dropout_layers[-1].output[0],None)

            self.dropout_logRegr_layer = LogisticRegression(
                input=input_logregr,
                n_in=hidden_layers_sizes[-1],
                n_out=n_out
            )
            self.dropout_layers.append(self.dropout_logRegr_layer)
            self.dropout_params.extend(self.dropout_logRegr_layer.params)
            
            self.logRegressionLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=self.hidden_layers_sizes[-1], n_out=n_out,
                         W=self.dropout_logRegr_layer.W, b=self.dropout_logRegr_layer.b)

        else:
            self.logRegressionLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_out
            )
        self.sigmoid_layers.append(self.logRegressionLayer)
        self.sigmoid_params.extend(self.logRegressionLayer.params)


        # Regularization terms
         
        for i in xrange(self.n_layers):
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
            self.L1 += abs(self.sigmoid_layers[i].W).sum()
            self.L2_sqr += (self.sigmoid_layers[i].W ** 2).sum()
         
        self.L1 = (self.L1 + abs(self.logRegressionLayer.W).sum())
        self.L2_sqr = (self.L2_sqr + (self.logRegressionLayer.W ** 2).sum())
        
        
        # Outputs

        if dropout_rates!=None:
            if self.x_NN!=None:
                 self.dropout_negative_log_likelihood = (
                     self.dropout_logRegr_layer.negative_log_likelihood(self.y, self.g, self.wij, self.knn)
                 )
            else:
                self.dropout_negative_log_likelihood = (
                    self.dropout_logRegr_layer.negative_log_likelihood(self.y)
                )
        else:
            if self.x_NN!=None:
                self.negative_log_likelihood = (
                    self.logRegressionLayer.negative_log_likelihood(self.y, self.g, self.wij, self.knn)
                )
            else:
                self.negative_log_likelihood = (
                    self.logRegressionLayer.negative_log_likelihood(self.y)
                )
        self.errors = ( 
            self.logRegressionLayer.errors(self.y)
        )

        if dropout_rates!=None:
            self.params.extend(self.dropout_params)
        else:
            self.params.extend(self.sigmoid_params)
        self.layers=self.sigmoid_layers



    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, test_shared_xy, learning_rate_init, cost_function, batch_size, mom_params, squared_filter_length_limit=10.0, train_set_x_NN_shared=None, gamma=None):
    
        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy
        (train_set_NN, wij_train)  = train_set_x_NN_shared
        
        

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        if test_shared_xy!=(None,None):
            (test_set_x,  test_set_y)  = test_shared_xy
            n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size

        if (train_set_NN!=None) and (wij_train!=None):
            n_train_NN_batches = train_set_NN.get_value(borrow=True).shape[0]/(batch_size*self.knn)
            

        mom_start = mom_params["start"]
        mom_end = mom_params["end"]
        mom_epoch_interval = mom_params["interval"]

        #Symbolic and shared variables
        index = T.lscalar('index')  # index to a [mini]batch
        epoch = T.scalar('epoch')
        learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)

        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost_function, param) for param in self.params]
        gparams_mom = []
    
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                            dtype=theano.config.floatX))      #broadcastable=param.broadcastable???????
            gparams_mom.append(gparam_mom)

        # Compute momentum for the current epoch
        mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

        # Update the step direction using momentum
        updates = OrderedDict()
        for gparam_mom, gparam in zip(gparams_mom, gparams):
            # Hinton's dropout paper
            updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

        # ... and take a step along that direction
        for param, gparam_mom in zip(self.params, gparams_mom):
            # we have included learning_rate in gparam_mom
            stepped_param = param + updates[gparam_mom]

            if (param.get_value(borrow=True).ndim == 2 and squared_filter_length_limit!=0):
                # constrain the norms of the COLUMNs of the weight, according to
                # https://github.com/BVLC/caffe/issues/109
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        if test_shared_xy!=(None,None): 
            test_model = theano.function(
                inputs=[index],
                outputs=self.errors,
                givens={
                    self.x: test_set_x[index * batch_size:(index + 1) * batch_size],
                    self.y: (test_set_y.flatten())[index * batch_size:(index + 1) * batch_size]
                }
#                on_unused_input='warn'
            )
#        theano.printing.pydotprint(test_model, outfile="/home/users/gchalk/test_file.png", var_with_name_simple=True)


        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                self.y: (valid_set_y.flatten())[index * batch_size:(index + 1) * batch_size]
            }
#            on_unused_input='warn'
        )
#        theano.printing.pydotprint(validate_model, outfile="/home/users/gchalk/validate_file.png", var_with_name_simple=True)
        

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model_NN=None
        if (train_set_NN!=None and gamma!=None and wij_train!=None):
            batch_size_NN = self.knn*batch_size 
            gamma_sh = theano.shared(numpy.cast[theano.config.floatX](gamma), name='gamma_sh', borrow=True)
           
            train_model_NN = theano.function(
                inputs=[index, epoch, theano.Param(learning_rate, default = learning_rate_init)],
                outputs=cost_function,
                updates=updates,
                givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    self.y: (train_set_y.flatten())[index * batch_size: (index + 1) * batch_size],
                    self.x_NN: train_set_NN, #[index * batch_size_NN: (index + 1) * batch_size_NN],
                    self.g: gamma_sh,            
                    self.wij: wij_train #[index * batch_size_NN: (index + 1) * batch_size_NN]
                }
#		on_unused_input = 'warn'
            )
        else:
            train_model = theano.function(
                inputs=[index, epoch, theano.Param(learning_rate, default = learning_rate_init)],
                outputs=cost_function,
                updates=updates,
                givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    self.y: (train_set_y.flatten())[index * batch_size: (index + 1) * batch_size]
                }
#                on_unused_input = 'warn'
            )
#        theano.printing.pydotprint(train_model, outfile="/home/users/gchalk/train_file.png", var_with_name_simple=True)

        train_model_error = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: (train_set_y.flatten())[index * batch_size: (index + 1) * batch_size],
            }
#            on_unused_input = 'warn'
        )
        

        if train_model_NN!=None:
            train_model = train_model_NN

        if test_shared_xy!=(None,None): 
            return test_model, validate_model, train_model, train_model_error
        else:
            return None, validate_model, train_model, train_model_error



    def write_model_to_raw(self, file_path):
        # output the model to tmp_path; this format is readable by PDNN
        if self.dropout_rates!=None:
            _nnet2file(self.layers, filename=file_path, input_factor = self.dropout_rates[0], factor = self.dropout_rates) ##or dropout_rates[1]?
        else:
            _nnet2file(self.layers, filename=file_path)



    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        output_layer_number = -1;
        for layer_index in range(1, len(self.hidden_layers_sizes) - 1):
            cur_layer_size = self.hidden_layers_sizes[layer_index]
            prev_layer_size = self.hidden_layers_sizes[layer_index-1]
            next_layer_size = self.hidden_layers_sizes[layer_index+1]
            if cur_layer_size < prev_layer_size and cur_layer_size < next_layer_size:
                output_layer_number = layer_index+1; break

        layer_number = len(self.layers)
        if output_layer_number == -1:
            output_layer_number = layer_number

        fout = smart_open(file_path, 'wb')
        for i in xrange(output_layer_number):
            activation_text = '<' + self.cfg.activation_text + '>' 
            if i == (layer_number-1) and with_softmax:   # we assume that the last layer is a softmax layer
                activation_text = '<softmax>'
            W_mat = self.layers[i].W.get_value()
            b_vec = self.layers[i].b.get_value()
            input_size, output_size = W_mat.shape
            W_layer = []; b_layer = ''
            for rowX in xrange(output_size):
                W_layer.append('')

            for x in xrange(input_size):
                for t in xrange(output_size):
                    W_layer[t] = W_layer[t] + str(W_mat[x][t]) + ' '

            for x in xrange(output_size):
                b_layer = b_layer + str(b_vec[x]) + ' '

            fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
            fout.write('[' + '\n')
            for x in xrange(output_size):
                fout.write(W_layer[x].strip() + '\n')
            fout.write(']' + '\n')
            fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
            fout.write(activation_text + ' ' + str(output_size) + ' ' + str(output_size) + '\n') 
        fout.close()




  
