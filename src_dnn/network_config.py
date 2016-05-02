# Copyright 2014    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import sys
import theano
import theano.tensor as T
from data_io import read_data_args, read_dataset

class NetworkConfig():

    def __init__(self):

        self.model_type = 'DNN'
        self.activation_text = 'sigmoid'
        self.activation = T.nnet.sigmoid
        self.do_maxout = False
        self.pool_size = 1
        self.do_dropout = False
        self.dropout_factor = []
        self.input_dropout_factor = 0.0
        self.max_col_norm = None
        self.l1_reg = None
        self.l2_reg = None

        # data reading
        self.train_sets = None
        self.train_xy = None
        self.train_x = None
        self.train_y = None

	self.train_xNN_sets = None  
	self.train_xNN_xy = None    
	self.train_xNN = None 
	self.wij_train = None

        self.valid_sets = None
        self.valid_xy = None
        self.valid_x = None
        self.valid_y = None

        self.test_sets = None
        self.test_xy = None
        self.test_x = None
        self.test_y = None

        # specifically for DNN
        self.n_ins = 0
        self.hidden_layers_sizes = []
        self.n_outs = 0
        self.non_updated_layers = []

        # number of epochs between model saving (for later model resuming)
        self.model_save_step = 1

        # the path to save model into Kaldi-compatible format
        self.cfg_output_file = ''
        self.param_output_file = ''
        self.kaldi_output_file = ''

# initialize pfile reading. TODO: inteference *directly* for Kaldi feature and alignment files
    def init_data_reading(self, train_data_spec, valid_data_spec, batch_size=None):
        train_dataset, train_dataset_args = read_data_args(train_data_spec)
        valid_dataset, valid_dataset_args = read_data_args(valid_data_spec)

        self.train_sets, self.train_xy, self.train_x, self.train_y = read_dataset(train_dataset, train_dataset_args, 'cpu', None, batch_size,False)
        self.valid_sets, self.valid_xy, self.valid_x, self.valid_y = read_dataset(valid_dataset, valid_dataset_args, 'cpu', None, batch_size,False)
        

    def init_data_manifold_reading(self, train_data_spec, knn=None, batch_size=None, manifold=None):   
        train_dataset, train_dataset_args = read_data_args(train_data_spec)
        self.train_xNN_sets, self.train_xNN_xy, self.train_xNN, self.wij_train = read_dataset(train_dataset, train_dataset_args, 'cpu', knn, batch_size, manifold)
        

    def init_data_reading_test(self, data_spec,batch_size):
        dataset, dataset_args = read_data_args(data_spec)
        self.test_sets, self.test_xy, self.test_x, self.test_y = read_dataset(dataset, dataset_args,'gpu',None,batch_size,False)
