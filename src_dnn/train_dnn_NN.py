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
from model_io import string_2_array, _file2nnet, _nnet2file, _cfg2file
from io_func import smart_open

#### rectified linear unit                                                                                                                  
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

def log_model(log_output_dir,
              train_data_spec,valid_data_spec,test_data_spec,manifold_data_spec,
              cost_path,valid_path,test_path,
              kaldi_output_file,param_output_file_pdnn,
              hidden_layers_sizes,n_in,n_out,activations,
              learning_rate_init,exponential_decay,n_epochs,batch_size,
              dropout_rates,knn,gamma,ptr_file):

    log_file = open(log_output_dir+"log_"+str(datetime.now())+".txt",'w+')
    sys.stderr.write("Log file created at " + "[" + str(datetime.now()) + "] " + ".\n")
    log_file.write("=======================================================\n")
    log_file.write("\n")
    log_file.write("\n")
    log_file.write("Log file created at "  + "[" + str(datetime.now()) + "] " + ".\n")
    log_file.write("\n")
    log_file.write("========== Input and output parameters. ==========\n")
    log_file.write("\n")
    log_file.write("Training dataset is: " + train_data_spec + ".\n")
    log_file.write("Validation dataset is: " + valid_data_spec + ".\n")
    if test_data_spec!=None:
        log_file.write("Testing dataset is: " + test_data_spec + ".\n")
    if manifold_data_spec==None:
        log_file.write("Baseline system.\n")
    else:
        log_file.write("Manifold regularized system. The manifold description is in: " + manifold_data_spec + ".\n")
        log_file.write("The total number of neighbours for each coordinate patch is: " + str(knn) + ".\n")
        log_file.write("The regularization weight is: " + str(gamma) + ".\n")
        if ptr_file!=None:
            log_file.write("The weights of the regularized system were initialized using the baseline system weights from file: " + ptr_file + ".\n\
")

    log_file.write("=======================================================\n")
    log_file.write("\n")
    log_file.write("\n")
    if cost_path!=None:
        log_file.write("The evolution of the cost per epoch is saved in: " + cost_path + ".\n")
    if valid_path!=None:
        log_file.write("The evolution of the error rate on the validation set per epoch is saved in: " + valid_path + ".\n")
    if test_path!=None:
        log_file.write("The evolution of the error rate on the test set per epoch is saved in: " + test_path + ".\n")
    if kaldi_output_file!=None:
        log_file.write("The Kaldi neural net model is saved in: " + kaldi_output_file + ".\n")
    if param_output_file_pdnn!=None:
        log_file.write("The neural net weights are saved in PDNN format in: " + param_output_file_pdnn + ".\n")
    log_file.write("=======================================================\n")
    log_file.write("\n")
    log_file.write("\n")
    log_file.write("========== Architecture and training parameters. ==========\n")
    log_file.write("\n")
    log_file.write("The sizes of the hidden layers are: " + str(hidden_layers_sizes) + ".\n")
    log_file.write("The input dimension is: " + str(n_in) + " and the output dimension is: " + str(n_out) + ".\n")
    log_file.write("The activation functions in the hidden layers are: " + activations + ".\n")
    log_file.write("The initial learning rate is: " + str(learning_rate_init) + ".\n")
    if exponential_decay==True:
        log_file.write("The learning rate is decaying exponentially per epoch.\n")
    else:
        log_file.write("The learning rate is is halved each time the validation error rate rises.\n")
    log_file.write("The number of training epochs is: " + str(n_epochs) + ".\n")
    log_file.write("The minibatch size is: "+ str(batch_size)+ ".\n")
    if dropout_rates!=None:
        log_file.write("Dropout regularization is implemented, with the following rate per layer, beginning with the input: " + str(dropout_rates) + ".\n")
    if squared_filter_length_limit!=0:
        log_file.write("The maximum squared Frobenius norm of the weight matrices per column is: " + str(squared_filter_length_limit) + ".\n")
    
    log_file.write("\n")
    log_file.write("\n")
    log_file.close()
 
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
        batch_index=0 
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
    
    while (not train_sets.is_finish()):
        train_sets.load_next_partition(train_xy)
        batch_index=0   
        train_sets_NN.load_next_partition(train_xy_NN)
        train_cost.append(train_fn(index=batch_index, epoch=epoch_cnt, learning_rate=learning_rate))
        train_error.append(train_error_fn(index=batch_index))
      
    train_sets.initialize_read()
    train_sets_NN.initialize_read()
    return (train_cost,train_error)



if __name__ == '__main__':

    #I/O
    rdir = "/local/gchalk/"
    train_data_spec = rdir+"train_si84_2kshort_tr95_3phone.pfile.1.gz" 
    valid_data_spec = rdir+"train_si84_2kshort_cv05_3phone.pfile.1.gz" 
    test_data_spec  = None
    manifold_data_spec = rdir + "manifold.pfile.tr95_LPDA30_phonesStates" 
    
    wdir = "/local/gchalk/outputs/"
    log_output_dir = "/home/users/gchalk/"
    param_path=None #wdir +"dnn_params"                                                                                                   
    cost_path=wdir+"obj_function_termi1_new1May"
    test_path=None #wdir+"error_function"                                                                                            
    valid_path=wdir+"valid_function_termi1_new1May"          
    best_params_cpu_path=None #wdir+"best_params_cpu"                                                                                      
    final_params_cpu_path=None #wdir+"final_params_cpu_30"                                                                                 
    kaldi_output_file=wdir+"kaldi_termi1_new1May.nnet"
    cfg_output_file=None #wdir+"cfg_manifold_base.cfg" 
    param_output_file_pdnn=wdir+"pdnn_manif1_1May.params"

    #Architecture
    hidden_layers_sizes=[1024,1024,1024,1024,1024]
    n_in=117
    n_out=1681 #pdf-id
    #total number of NNs on manifold
    knn=60 
    activations=[T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid]

    #Training policy
    learning_rate_init=0.1
    exponential_decay=True
    learning_rate_decay=0.15
    
    L1_reg=0.000
    L2_reg=0.000
    gamma=0.00000000000000000001 
    
    n_epochs=15
    n_epochs_reduce_dropout_pretrain=30
    n_epochs_start_dropout_finetune=n_epochs_reduce_dropout_pretrain + 5
    mom_params = {"start": 0.99, "end": 0.99, "interval": 5}
    batch_size=256

    dropout_rates=None #[0.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    squared_filter_length_limit=15
    
    ptr_layer_number = 6
    ptr_file = wdir + "pdnn_manif1_.params"  # "pdnn_base_Init.params"
    

    if (param_path!=None):
        save_file = open(param_path,'wb')
    if (test_path!=None):
        save_error_test = open(test_path,'wb')
    if (cost_path!=None):
        save_cost = open(cost_path,'wb')
    if (valid_path!=None):
        save_error_valid = open(valid_path,'wb')
    if (final_params_cpu_path!=None):
    	final_params_cpu = open(final_params_cpu_path,'wb')
    

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig()
    cfg.init_data_reading(train_data_spec, valid_data_spec, batch_size)    
    
    if manifold_data_spec!=None:
	cfg.init_data_manifold_reading(manifold_data_spec, knn=knn, batch_size=batch_size, manifold=True)  

    if test_data_spec!=None:
       cfg.init_data_reading_test(test_data_spec,batch_size)

     
    if activations[0] == ReLU:
        cfg.activation_text = 'ReLU'
    elif activations[0] == T.tanh:
        cfg.activation_text = 'Tanh'
    else:
        cfg.activation_text = 'Sigmoid'


    log_model(log_output_dir,train_data_spec,valid_data_spec,test_data_spec,manifold_data_spec,cost_path,valid_path,test_path,kaldi_output_file,param_output_file_pdnn,hidden_layers_sizes,n_in,n_out,cfg.activation_text,learning_rate_init,exponential_decay,n_epochs,batch_size,dropout_rates,knn,gamma,ptr_file)

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = theano.tensor.shared_randomstreams.RandomStreams(numpy_rng.randint(2 ** 30))
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](learning_rate_init), name='lr', borrow=True)

    ######################
    # BUILD MODEL #
    ######################

    log('> ... building the model')

    #setup model
    dnn = MLP(numpy_rng=numpy_rng, 
              input=(cfg.train_x,cfg.train_xNN), 
              n_in=n_in,
              hidden_layers_sizes=hidden_layers_sizes, 
              n_out=n_out,
              knn=knn,
              dropout_rates=dropout_rates,
              activations=activations,
              cfg=cfg)


    # load pre-trained parameters
    if (ptr_layer_number > 0) and (ptr_file != None):
        log('> ... loading pre-trained weight matrices')
        _file2nnet(dnn.layers, set_layer_num = ptr_layer_number, filename = ptr_file)


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
    while (learning_rate != 0) and (epoch_counter < n_epochs) :
        
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
            if (best_params_cpu_path!=None):
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
    

    if param_output_file_pdnn != None:
       if dropout_rates != None: 
           _nnet2file(dnn.layers, filename=param_output_file_pdnn, input_factor = dropout_rates[0], factor = dropout_rates[1:])
       else:
           _nnet2file(dnn.layers, filename=param_output_file_pdnn)
       log('> ... the final PDNN model parameter is ' + param_output_file_pdnn)

    if cfg_output_file != None:
        _cfg2file(dnn.cfg, filename=cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg_output_file)

   
    # output the model into Kaldi-compatible format
    if kaldi_output_file != None:
        dnn.write_model_to_kaldi(kaldi_output_file)
        log('> ... the final Kaldi model is ' + kaldi_output_file) 

    if (param_path!=None):
        for param in dnn.params:
            save_parameters(param.get_value(),save_file)            
        save_file.close() 
    if (cost_path!=None):
        save_parameters(cost_evolution,save_cost)
        save_cost.close()
    if (valid_path!=None):
        save_parameters(valid_evolution,save_error_valid)
        save_error_valid.close()
    if (test_path!=None):
        save_parameters(test_evolution,save_error_test)
        save_error_test.close()
    if (final_params_cpu_path!=None):
	for param in dnn.params:
        	cPickle.dump(param.get_value(), final_params_cpu , -1) 
    	final_params_cpu.close()
    

    print(('Optimization complete.'))
    print >> sys.stderr, ('The code for file ' 
                          + os.path.split(__file__)[1] 
                          + ' ran for %.2fm' % ((end_time - start_time) / 60.))

