# Copyright 2013    Yajie Miao    Carnegie Mellon University
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

import theano.tensor as T
from io_func import smart_open

def string2bool(string):
    return len(string) > 0 and string[0] in 'TtYy1'     # True, true, Yes, yes, 1

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args
"""
def parse_ignore_label(ignore_string):
    ignore_set = set()
    for x in ignore_string.split(':'):
        if '-' in x:
            start, end = (int(y) for y in x.split('-'))
            ignore_set.update(range(start, end + 1))
        else:
            ignore_set.add(int(x))
    return ignore_set

def parse_map_label(map_string):
    map_dict = {}
    for x in map_string.split('/'):
        source, target = x.split(':')
        target = int(target)
        if '-' in source:
            start, end = (int(y) for y in source.split('-'))
            for i in range(start, end + 1):
                map_dict[i] = target
        else:
            map_dict[int(source)] = target
    return map_dict

def parse_lrate(lrate_string):
    elements = lrate_string.split(":")
    # 'D:0.08:0.5:0.05,0.05:15'
    if elements[0] == 'D':  # NewBob
        if (len(elements) != 5):
            return None
        values = elements[3].split(',')
        lrate = LearningRateExpDecay(start_rate=float(elements[1]),
                                 scale_by = float(elements[2]),
                                 min_derror_decay_start = float(values[0]),
                                 min_derror_stop = float(values[1]),
                                 init_error = 100,
                                 min_epoch_decay_start=int(elements[4]))
        return lrate

    # 'C:0.08:15'
    if elements[0] == 'C':  # Constant
        if (len(elements) != 3):
            return None
        lrate = LearningRateConstant(learning_rate=float(elements[1]),
                                 epoch_num = int(elements[2]))
        return lrate

    # 'MD:0.08:0.5:0.05,0.0002:8'
    if elements[0] == 'MD':  # Min-rate NewBob
        if (len(elements) != 5):
            return None
        values = elements[3].split(',')
        lrate = LearningMinLrate(start_rate=float(elements[1]),
                                 scale_by = float(elements[2]),
                                 min_derror_decay_start = float(values[0]),
                                 min_lrate_stop = float(values[1]),
                                 init_error = 100,
                                 min_epoch_decay_start=int(elements[4]))
        return lrate

    # 'FD:0.08:0.5:10,6'
    if elements[0] == 'FD':  # Min-rate NewBob
        if (len(elements) != 4):
            return None
        values = elements[3].split(',')
        lrate = LearningFixedLrate(start_rate=float(elements[1]),
                                 scale_by = float(elements[2]),
                                 decay_start_epoch = int(values[0]),
                                 stop_after_deday_epoch = int(values[1]),
                                 init_error = 100)
        return lrate

    # 'A:0.1:1.00,1.05:1.04,0.7:1.00,6:100'
    if elements[0] == 'A':  # Adaptive
        if len(elements) != 6:
            return None
        lrate = LearningRateAdaptive(lr_init = float(elements[1]),
                                     thres_inc = float(elements[2].split(',')[0]),
                                     factor_inc = float(elements[2].split(',')[1]),
                                     thres_dec = float(elements[3].split(',')[0]),
                                     factor_dec = float(elements[3].split(',')[1]),
                                     thres_fail = float(elements[4].split(',')[0]),
                                     max_fail = float(elements[4].split(',')[1]),
                                     max_epoch = int(elements[5]))
        return lrate
"""

def parse_activation(act_str):
    if act_str == 'linear':
        return lambda x: x
    if act_str == 'sigmoid':
        return T.nnet.sigmoid
    if act_str == 'tanh':
        return T.tanh
    if act_str == 'rectifier' or act_str == 'relu':
        return lambda x: T.maximum(0.0, x)
    return T.nnet.sigmoid

def activation_to_txt(act_func):
    if act_func == T.nnet.sigmoid:
        return 'sigmoid'
    if act_func == T.tanh:
        return 'tanh'



