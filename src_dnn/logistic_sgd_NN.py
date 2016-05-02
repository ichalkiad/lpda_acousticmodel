
"""
Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T


def log_softmax(q):
    rebased_q = q - q.max(axis=1).dimshuffle(0,'x')
    return rebased_q - T.log(T.exp(rebased_q).sum(axis=1)).dimshuffle(0,'x')


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,W=None,b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
    

        self.type = 'fc'
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
	if W is None:
           W = theano.shared(
              value=numpy.zeros(
                  (n_in, n_out),
                  dtype=theano.config.floatX
              ),
              name='W',
              borrow=True
           )
        # initialize the biases b as a vector of n_out 0s
        if b is None:
	   b = theano.shared(
              value=numpy.zeros(
                  (n_out,),
                  dtype=theano.config.floatX
              ),
              name='b',
              borrow=True
           )

        self.W = W
        self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
 
        self.p_y_given_x = T.nnet.softmax(T.dot(input[0], self.W) + self.b)
        self.output = self.p_y_given_x

        if input[1]!=None:
             self.p_y_given_x_NN = T.nnet.softmax(T.dot(input[1], self.W) + self.b)
        else:
            self.p_y_given_x_NN = None

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def predictions(self):
        return self.p_y_given_x
	
    def negative_log_likelihood(self, y, gamma=None, wij=None, knn=None):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        self.p_y_given_x = T.clip(self.p_y_given_x,0.000001, 0.999999)
       
        #Negative log-likelihood cost
        x_entr = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        
     
        if (gamma!=None) and (wij!=None) and (self.p_y_given_x_NN!=None) and (knn!=None):
            self.p_y_given_x_NN = T.clip(self.p_y_given_x_NN,0.00000001, 0.99999999)
            #Manifold regularization
            y_repeated = T.extra_ops.repeat(self.p_y_given_x,knn,axis=0)
            manifold_term = T.sum( T.reshape( (T.sum((y_repeated - self.p_y_given_x_NN)**2,1)*wij ), (self.p_y_given_x.shape[0],knn)) ,1) 

            return T.mean(-x_entr + gamma*manifold_term)  
        else:
            return -T.mean(x_entr)
            


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        if (data_y != None):
            shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        else:
            return shared_x, None
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

   


