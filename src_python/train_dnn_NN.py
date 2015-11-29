import cPickle
import gzip
import os
import sys
import time
from datetime import datetime
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import timeit

from mlp_dropout_oneNet_NN import MLP
from network_config import NetworkConfig


#### rectified linear unit                                                                                                                      
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

def save_parameters(params,save_file):
    cPickle.dump(params, save_file, -1)  


# validation on the valid data; this involves a forward pass of all the valid data into the network,
# mini-batch by mini-batch
# valid_fn: the compiled valid function
# valid_sets: the dataset object for valid
# valid_xy: the tensor variables for valid dataset
# batch_size: the size of mini-batch
# return: a list containing the *error rates* on each mini-batch
def validate_by_minibatch(valid_fn, cfg, batch_size):
    valid_sets = cfg.valid_sets; valid_xy = cfg.valid_xy
    valid_error = []
    while (not valid_sets.is_finish()):
        valid_sets.load_next_partition(valid_xy)
        for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
            valid_error.append(valid_fn(index=batch_index))
    valid_sets.initialize_read()
    return valid_error


def test_by_minibatch(test_fn, cfg, batch_size):
    test_sets = cfg.test_sets; test_xy = cfg.test_xy
    test_error = []
    while (not test_sets.is_finish()):
        test_sets.load_next_partition(test_xy)
        for batch_index in xrange(test_sets.cur_frame_num / batch_size):  # loop over mini-batches
            test_error.append(test_fn(index=batch_index))
    test_sets.initialize_read()
    return test_error

# one epoch of mini-batch based SGD on the training data
# train_fn: the compiled training function
# train_sets: the dataset object for training
# train_xy: the tensor variables for training dataset
# batch_size: the size of mini-batch
# learning_rate: learning rate
# momentum: momentum
# return: a list containing the *error rates* on each mini-batch
def train_sgd(train_fn, train_error_fn, cfg, batch_size, learning_rate, epoch_cnt):
    train_sets = cfg.train_sets; train_xy = cfg.train_xy
    train_error = []
    train_cost = []
    while (not train_sets.is_finish()):
        train_sets.load_next_partition(train_xy)
        for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
            train_cost.append(train_fn(index=batch_index, epoch=epoch_cnt, learning_rate=learning_rate))
            train_error.append(train_error_fn(index=batch_index))
    train_sets.initialize_read()
    return (train_cost,train_error)

def train_sgd_manifold(train_fn, train_error_fn, cfg, batch_size, learning_rate, epoch_cnt):
    train_sets = cfg.train_sets; train_xy = cfg.train_xy
    train_sets_NN = cfg.train_xNN_sets; train_xy_NN = cfg.train_xNN_xy
    train_error = []
    train_cost = []
    set_finish = 1
    while (not train_sets.is_finish() and set_finish==1):
        train_sets.load_next_partition(train_xy)
        for batch_index in xrange(3100):  #train_sets.cur_frame_num / batch_size):  #supposed to be the same as train_sets_NN.cur_frame_num/(batch_size*NN_num)   	    
	    train_sets_NN.load_next_partition(train_xy_NN)
            train_cost.append(train_fn(index=batch_index, epoch=epoch_cnt, learning_rate=learning_rate))
            train_error.append(train_error_fn(index=batch_index))

        set_finish=0
      

    train_sets.initialize_read()
    train_sets_NN.initialize_read()
    return (train_cost,train_error)



if __name__ == '__main__':

    train_data_spec = "./DNN_py/dnn_pfiles/train_si84_2kshort_tr95.pfile.1.gz" 
    valid_data_spec = "./DNN_py/dnn_pfiles/train_si84_2kshort_cv05.pfile.1.gz"
    manifold_data_spec = "./DNN_py/dnn_pfiles/manifold.pfile.1.tr95" 

    test_data_spec  = None
    wdir = "./DNN_py/dnn_pfiles/"

    learning_rate_init=0.01
    exponential_decay=True
    learning_rate_decay=0.01
    L1_reg=0.000
    L2_reg=0.000
    n_epochs=5
    n_epochs_reduce_dropout_pretrain=30
    n_epochs_start_dropout_finetune=n_epochs_reduce_dropout_pretrain + 10
    mom_params = {"start": 0.8, "end": 0.99, "interval": 30}
    batch_size=256
    hidden_layers_sizes=[1024, 1024, 1024, 1024]
    gamma=0.0000001 #,0.0009,0.03
    n_in=117
    n_out=132
    knn=28 #total number of NNs on manifold
    dropout_rates=[0.0, 0.4, 0.4, 0.4, 0.4]
    activations=[ReLU, ReLU, ReLU, ReLU]
    squared_filter_length_limit=10.0
    param_path="./DNN_py/dnn_pfiles/dnn_params"
    cost_path="./DNN_py/dnn_pfiles/obj_functionR"
    test_path=None  #"./DNN_py/dnn_pfiles/error_function"
    valid_path="./DNN_py/dnn_pfiles/valid_functionR"
    best_params_cpu_path="./DNN_py/dnn_pfiles/best_params_cpu"
    kaldi_output_file='./DNN_py/dnn_pfiles/kaldi_nnetR.nnet'
    final_params_cpu_path="./DNN_py/dnn_pfiles/final_params_cpu"


    if (param_path!=None):
        save_file = open(param_path,'wb')
    if (test_path!=None):
        save_error_test = open(test_path,'wb')
    save_cost = open(cost_path,'wb')
    save_error_valid = open(valid_path,'wb')
    final_params_cpu = open(final_params_cpu_path,'wb')
    

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig()
    cfg.init_data_reading(train_data_spec, valid_data_spec)    
    
    if manifold_data_spec!=None:
	cfg.init_data_manifold_reading(manifold_data_spec)  

    if test_data_spec!=None:
       cfg.init_data_reading_test(test_data_spec)
     
    if activations[0] == ReLU:
        cfg.activation_text = 'ReLU'
    elif activations[0] == T.tanh:
        cfg.activation_text = 'Tanh'
    else:
        cfg.activation_text = 'Sigmoid'

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = theano.tensor.shared_randomstreams.RandomStreams(numpy_rng.randint(2 ** 30))
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](learning_rate_init), name='lr', borrow=True)

    ######################
    # BUILD MODEL #
    ######################

    log('> ... building the model')

    # setup model
    dnn = MLP(numpy_rng=numpy_rng, 
              input=(cfg.train_x,cfg.train_xNN), 
              n_in=n_in,
              hidden_layers_sizes=hidden_layers_sizes, 
              n_out=n_out,
              knn=knn,
              dropout_rates=dropout_rates,
              activations=activations,
              cfg=cfg)

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    
    if dropout_rates==None:
        cost = (
            dnn.negative_log_likelihood
            + L1_reg * dnn.L1
            + L2_reg * dnn.L2_sqr
        )
    else:
        cost = (
            dnn.dropout_negative_log_likelihood
            + L1_reg * dnn.L1
            + L2_reg * dnn.L2_sqr
        )
    

    if (exponential_decay):
        epoch_cnt = T.scalar('epoch_cnt')
        decay_learning_rate = theano.function(inputs=[epoch_cnt], 
                                           outputs=learning_rate, 
                                           updates={learning_rate: learning_rate_init/(1+learning_rate_decay*epoch_cnt) }
                                          )
    else:
        decay_learning_rate = theano.function(inputs=[], 
                                          outputs=learning_rate, 
                                          updates={learning_rate: learning_rate/2 }
                                          )

    if (cfg.wij_train!=None):
	wij_t = T.cast(cfg.wij_train,theano.config.floatX)
    else:
	wij_t = None

    test_model, validate_model, train_model, train_model_error = dnn.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y),(cfg.test_x, cfg.test_y),learning_rate_init, cost, batch_size, mom_params,squared_filter_length_limit, (cfg.train_xNN, wij_t), gamma )

 
    ###############
    # TRAIN MODEL #
    ###############

    cv_error_drop = False
    best_validation_loss = numpy.inf
    test_score = 0.
    cost_evolution = []
    test_evolution = []
    valid_evolution = []
    epoch_counter = 0
    prev_validation_loss = 0.0
    start_time = timeit.default_timer()


    log('> ... finetuning the model')
    while (learning_rate != 0) and (epoch_counter < n_epochs):
        
        if ((epoch_counter % 2 == 0) and param_path!=None):
            for param in dnn.params:
                save_parameters(param.get_value(),save_file)

        epoch_counter = epoch_counter + 1

        if dropout_rates!=None:
            if (n_epochs_reduce_dropout_pretrain!=None) and (epoch_counter > n_epochs_reduce_dropout_pretrain):
                dnn.dropout_rates = [ x*0.4 for x in dnn.dropout_rates]
            if (n_epochs_start_dropout_finetune!=None) and (epoch_counter > n_epochs_start_dropout_finetune):
                dnn.dropout_rates = [ x*0.0 for x in dnn.dropout_rates]
    

        # one epoch of sgd training 
        if cfg.train_xNN!=None:
            (train_cost, train_error) = train_sgd_manifold(train_model, train_model_error, cfg, batch_size, learning_rate.get_value(borrow=True), epoch_counter)
        else:
            (train_cost, train_error) = train_sgd(train_model, train_model_error, cfg, batch_size, learning_rate.get_value(borrow=True), epoch_counter)

        cost_evolution.append(numpy.mean(train_cost))
        log('> epoch %d, training error %f ' % (epoch_counter, 100*numpy.mean(train_error)) + '(%)')
        # validation
        valid_error = validate_by_minibatch(validate_model, cfg, batch_size)
        this_validation_loss = numpy.mean(valid_error)
        valid_evolution.append(this_validation_loss)
        log('> epoch %d, lrate %f, validation error %f ' % (epoch_counter, learning_rate.get_value(borrow=True), 100*this_validation_loss) + '(%)')

        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            best_params_cpu  = open(best_params_cpu_path,'wb')
            for param in dnn.params:
                cPickle.dump(param.get_value(borrow=True), best_params_cpu , -1) 
            best_params_cpu.close()
            if (test_model!=None):
                test_error = test_by_minibatch(test_model, cfg, batch_size)
                test_score = numpy.mean(test_error)
                test_evolution.append(test_score)
                log('> epoch %d, lrate %f, test error %f ' % (epoch_counter, learning_rate.get_value(borrow=True), 100*test_score) + '(%)')
            cv_error_drop = True
        else:
            cv_error_drop = False
                
	if (this_validation_loss < prev_validation_loss):
	    cv_error_drop = True

        if (cv_error_drop==False):
            if (exponential_decay):
                decay_learning_rate(epoch_counter)
            else:
                decay_learning_rate()
        prev_validation_loss = this_validation_loss


    end_time = timeit.default_timer()
    
    

   
    # output the model into Kaldi-compatible format
    if kaldi_output_file != '':
        dnn.write_model_to_kaldi(kaldi_output_file)
        log('> ... the final Kaldi model is ' + kaldi_output_file) 

    if (param_path!=None):
        for param in dnn.params:
            save_parameters(param.get_value(),save_file)            
        save_file.close()            
    save_parameters(cost_evolution,save_cost)
    save_cost.close()
    save_parameters(valid_evolution,save_error_valid)
    save_error_valid.close()
    if (test_path!=None):
        save_parameters(test_evolution,save_error_test)
        save_error_test.close()
    for param in dnn.params:
        cPickle.dump(param.get_value(), final_params_cpu , -1) 
    final_params_cpu.close()
    


    print(('Optimization complete.'))
    print >> sys.stderr, ('The code for file ' 
                          + os.path.split(__file__)[1] 
                          + ' ran for %.2fm' % ((end_time - start_time) / 60.))

