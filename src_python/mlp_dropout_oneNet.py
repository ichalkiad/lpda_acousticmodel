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
import theano.printing
from theano.ifelse import ifelse
from collections import OrderedDict

from logistic_sgd import LogisticRegression, shared_dataset


#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


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
        self.input_x , self.input_xNN = input
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

        lin_output = T.dot(input[0], self.W) + self.b
        lin_output_NN = T.dot(input[1], self.W) + self.b
   
        self.output = (
            (lin_output,lin_output_NN) if activation is None
            else (activation(lin_output),activation(lin_output_NN))
        )

        # parameters of the model
        self.params = [self.W, self.b]



def _dropout_from_layer(rng, (x,x_NN), p):

    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=x.shape)
    mask_NN = srng.binomial(n=1, p=1-p, size=x_NN.shape)
    output = (x * T.cast(mask, theano.config.floatX),
                x_NN * T.cast(mask_NN, theano.config.floatX))
    
    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, dropout_rate, W=None, b=None,
                 activation=T.nnet.sigmoid ):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    """Multi-Layer Perceptron Class
    """

    def __init__(self, rng, input, n_in, hidden_layers_sizes, n_out,dropout_rates=None,activations=None):
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
       

        self.params = []

        if dropout_rates!=None:
            self.L1_dropout = theano.shared(numpy.cast[theano.config.floatX](0.0), name='L1_drop', borrow=True)
            self.L2_sqr_dropout = theano.shared(numpy.cast[theano.config.floatX](0.0), name='L2_sqr_drop', borrow=True)
            self.dropout_params = []
            self.dropout_layers = []
        else:
            self.L1 = theano.shared(numpy.cast[theano.config.floatX](0.0), name='L1', borrow=True)
            self.L2_sqr = theano.shared(numpy.cast[theano.config.floatX](0.0), name='L2_sqr', borrow=True)
            self.sigmoid_params = []
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
               if dropout_rates!=None:
                    dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[i])
               else:
                    layer_input = input
            else:
               if dropout_rates!=None:
                    dropout_layer_input =  (T.cast((1/(1-dropout_rates[i-1])), theano.config.floatX)*self.dropout_layers[i-1].output[0],T.cast((1/(1-dropout_rates[i-1])), theano.config.floatX)*self.dropout_layers[i-1].output[1])
               else:
                    layer_input = self.sigmoid_layers[i-1].output

            if activations!=None:
                 activation = activations[i]
            else:
                 activation = T.nnet.sigmoid


            # hidden layers for training network
            if dropout_rates!=None:
                dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=dropout_layer_input,
                    n_in=input_size,
                    n_out=hidden_layers_sizes[i],
                    activation=activation,
                    dropout_rate=dropout_rates[i]
                )
                self.dropout_layers.append(dropout_layer)
                self.dropout_params.extend(dropout_layer.params)
            else:
                sigmoid_layer = HiddenLayer(
                    rng=rng,
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
            self.dropout_logRegr_layer = LogisticRegression(
                input=self.dropout_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_out
            )
            self.dropout_params.extend(self.dropout_logRegr_layer.params)
        else:
            self.logRegressionLayer = LogisticRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_out
            )
            self.sigmoid_params.extend(self.logRegressionLayer.params)


        # Regularization terms
         
        for i in xrange(self.n_layers):
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
            if dropout_rates!=None: 
                self.L1_dropout += abs(self.dropout_layers[i].W).sum()
                self.L2_sqr_dropout += (self.dropout_layers[i].W ** 2).sum()
            else:
                self.L1 += abs(self.sigmoid_layers[i].W).sum()
                self.L2_sqr += (self.sigmoid_layers[i].W ** 2).sum()
         

        if dropout_rates!=None:
            self.L1_dropout = (self.L1_dropout + abs(self.dropout_logRegr_layer.W).sum())
            self.L2_sqr_dropout = (self.L2_sqr_dropout + (self.dropout_logRegr_layer.W ** 2).sum())
        else:
            self.L1 = (self.L1 + abs(self.logRegressionLayer.W).sum())
            self.L2_sqr = (self.L2_sqr + (self.logRegressionLayer.W ** 2).sum())
        
        
        # Outputs

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer

        if dropout_rates!=None:
            self.dropout_negative_log_likelihood = (
                self.dropout_logRegr_layer.negative_log_likelihood
            )
            self.errors = ( self.dropout_logRegr_layer.errors
            )
        else:
            self.negative_log_likelihood = (
                self.logRegressionLayer.negative_log_likelihood
            )
            self.errors = ( self.logRegressionLayer.errors
            )

        if dropout_rates!=None:
            self.params.extend(self.dropout_params)
        else:
            self.params.extend(self.sigmoid_params)



def save_parameters(params,save_file):
    cPickle.dump(params, save_file, -1)  



def load_dataset(i):
    path1 = "/home/users/gchalk/DNN_py/chunked_dataset/data" + repr(i) + ".h5"  
    path2 = "/home/users/gchalk/DNN_py/chunked_dataset/labels" + repr(i) + ".h5"
    path3 = "/home/users/gchalk/DNN_py/chunked_dataset/neighbors" + repr(i) + ".h5"
    path4 = "/home/users/gchalk/DNN_py/chunked_dataset/wij_diff" + repr(i) + ".h5"

    print '... loading dataset ' + repr(i)

    data = numpy.array(h5py.File(path1,"r")["data"])
    labels = numpy.array(h5py.File(path2,"r")["labels"]).transpose() ## #data X 1
    neighbors = numpy.array(h5py.File(path3,"r")["data"])
    wij = numpy.array(h5py.File(path4,"r")["Wij_diff"])

    
#    labels = labels - 1  # not when using pdf-ids
#    labels = labels.T.astype(int)
    
    train_set = (data,labels)
    train_set_x, train_set_y = shared_dataset(train_set)
    train_set_x_NN, train_set_y_NN = shared_dataset((neighbors,None))
    wij_sh = theano.shared(numpy.cast[theano.config.floatX](wij),name='wij_sh',borrow=True)

    return (train_set_x,train_set_y,train_set_x_NN,wij_sh)
    

def test_mlp(learning_rate_init=0.9,
             learning_rate_decay=0.01,
             L1_reg=0.000,
             L2_reg=0.000,
             n_epochs_set=1,
             n_epochs_per_chunk=50,
             mom_params = {"start": 0.7, "end": 0.99, "interval": 10},
             batch_size=320,
             hidden_layers_sizes=[600, 600, 600, 600, 600],
             gamma=0.000008,#0.0009,0.03
             dropout_rates=[0.0, 0.4 ,0.4, 0.4, 0.4, 0.4],
             activations=[ReLU, ReLU, ReLU, ReLU, ReLU],
             squared_filter_length_limit=10.0,
             path="/home/users/gchalk/DNN_py/dnn_params",
             cost_path="/home/users/gchalk/DNN_py/obj_function",
             test_path="/home/users/gchalk/DNN_py/error_function",
             valid_path="/home/users/gchalk/DNN_py/valid_function",
             params_cpu="/home/users/gchalk/DNN_py/best_params_cpu",
             num_datasets=10
             ):
    
    
#    save_file = open(path,'wb')
    save_cost = open(cost_path,'wb')
    save_error_test = open(test_path,'wb')
    save_error_valid = open(valid_path,'wb')
#    final_params_cpu = open(params_cpu,'wb')
    best_params_cpu  = open(    
   
    dataset_idx = 1  #dataset index
    (train_set_x,train_set_y,train_set_x_NN,wij_sh) = load_dataset(dataset_idx)

    test_data = numpy.array(h5py.File("/home/users/gchalk/DNN_py/chunked_dataset/mfcc_valid117D.h5","r")["mfcc_sampled"])
    test_labels = numpy.array(h5py.File("/home/users/gchalk/DNN_py/chunked_dataset/labels_valid.h5","r")["labels"])
    valid_data = numpy.array(h5py.File("/home/users/gchalk/DNN_py/chunked_dataset/mfcc_test117D.h5","r")["mfcc_sampled"])
    valid_labels = numpy.array(h5py.File("/home/users/gchalk/DNN_py/chunked_dataset/labels_test.h5","r")["labels"])

    test_set = (test_data,test_labels)
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set = (valid_data, valid_labels)
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    gamma_sh = theano.shared(numpy.cast[theano.config.floatX](gamma), name='gamma_sh', borrow=True)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    batch_size_NN = wij_sh.shape[1]*batch_size 

    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    x_NN = T.matrix('x_NN')
    g = T.scalar('g',dtype=theano.config.floatX)
    wij_diff = T.matrix('wij_diff')
    learning_rate = theano.shared(numpy.asarray(learning_rate_init, dtype=theano.config.floatX))
    rng = numpy.random.RandomState(1234)


    # construct the MLP class
    classifier = MLP(
          rng=rng,
          input=(x,x_NN),
          n_in=117,
	  hidden_layers_sizes=hidden_layers_sizes,
	  n_out=132,
          dropout_rates=dropout_rates,
          activations=activations
    )	  
	
    if dropout_rates==None:
        cost = (
            classifier.negative_log_likelihood(y, g, wij_diff)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
        )
    else:
        cost = (
            classifier.dropout_negative_log_likelihood(y, g, wij_diff)
            + L1_reg * classifier.L1_dropout
            + L2_reg * classifier.L2_sqr_dropout
        )
        
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: (test_set_y.flatten())[index * batch_size:(index + 1) * batch_size]
        },
#        on_unused_input='warn'
    )
#    theano.printing.pydotprint(test_model, outfile="/home/users/gchalk/test_file.png", var_with_name_simple=True)
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: (valid_set_y.flatten())[index * batch_size:(index + 1) * batch_size]
        },
#        on_unused_input='warn'
    )
#    theano.printing.pydotprint(validate_model, outfile="/home/users/gchalk/validate_file.png", var_with_name_simple=True)


    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    gparams_mom = []
    for param in classifier.params:
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
    for param, gparam_mom in zip(classifier.params, gparams_mom):
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

                       
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, epoch],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: (train_set_y.flatten())[index * batch_size: (index + 1) * batch_size],
            x_NN: train_set_x_NN[index * batch_size_NN: (index + 1) * batch_size_NN],
            g: gamma_sh,            
            wij_diff: wij_sh[index * batch_size: (index + 1) * batch_size]
        },
#	on_unused_input = 'warn'
    )
#    theano.printing.pydotprint(train_model, outfile="/home/users/gchalk/train_file.png", var_with_name_simple=True)

#    decay_learning_rate = theano.function(inputs=[epoch], 
#                                          outputs=learning_rate, 
#                                          updates={learning_rate: learning_rate_init/(1+learning_rate_decay*epoch) }
#                                          )
    decay_learning_rate = theano.function(inputs=[], 
                                          outputs=learning_rate, 
                                          updates={learning_rate: learning_rate/2 }
                                          )

    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # go through this many minibatches before checking the network on the validation set;
    # in this case we check every epoch
    validation_frequency = n_train_batches

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    cost_evolution = []
    test_evolution = []
    valid_evolution = []
    epoch_counter = 0
    done_looping = False
    n_epochs_init = n_epochs_per_chunk    
    prev_validation_loss = 0.0

    for l in xrange(n_epochs_set):


     while (epoch_counter < n_epochs_per_chunk) and (not done_looping):

#        if (epoch_counter % 50 == 0):
#            save_parameters(classifier.params,save_file)

        
        epoch_counter = epoch_counter + 1
        	
	for minibatch_index in xrange(n_train_batches):
             
            minibatch_avg_cost = train_model(minibatch_index, epoch_counter)
            cost_evolution.append(minibatch_avg_cost)
            
            # iteration number
            iter = (epoch_counter - 1) * n_train_batches + minibatch_index
            # validate at the end of each epoch
            if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

		valid_evolution.append(this_validation_loss)

                print(
                    'epoch %i, minibatch %i/%i, learning rate %f, validation error %f %%' %
                    (
                     epoch_counter,
                     minibatch_index + 1,
                     n_train_batches,
                     learning_rate.get_value(borrow=True),
                     this_validation_loss * 100.
                    ) 
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    for param in classifier.params:
                        cPickle.dump(param.get_value(borrow=True), best_params_cpu , -1) 

                    # test it on the test set
                    test_losses = [test_model(i) for i
                              in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

		    test_evolution.append(test_score)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                          'best model %f %%') %
                          (epoch_counter, minibatch_index + 1, n_train_batches,
                          test_score * 100.))
                    cv_error_drop = True
                else:
                    cv_error_drop = False
                
		if (this_validation_loss < prev_validation_loss):
		    cv_error_drop = True

                if (cv_error_drop==False):
            	       decay_learning_rate()
#                print learning_rate.get_value(borrow=True)
		prev_validation_loss = this_validation_loss


        if (epoch_counter == n_epochs_per_chunk):
	    if (dataset_idx != num_datasets):
            	dataset_idx = dataset_idx + 1
	    else:
		break
            n_epochs_per_chunk += n_epochs_init 
#            save_parameters(classifier.params,save_file) 
            (train_set_x,train_set_y,train_set_x_NN,wij_sh) = load_dataset(dataset_idx)   
     
            
             

     n_epochs_per_chunk += n_epochs_init 
     dataset_idx=1
     (train_set_x,train_set_y,train_set_x_NN,wij_sh) = load_dataset(dataset_idx)
     
	

    end_time = timeit.default_timer()

#    save_parameters(classifier.params,save_file)            
#    save_file.close()            
    save_parameters(cost_evolution,save_cost)
    save_cost.close()
    save_parameters(valid_evolution,save_error_valid)
    save_error_valid.close()
    save_parameters(test_evolution,save_error_test)
    save_error_test.close()
#    for param in classifier.params:
#        cPickle.dump(param.get_value(borrow=True), final_params_cpu , -1) 
#    final_params_cpu.close()
    best_params_cpu.close()

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
