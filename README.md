# Manifold-Regularized DNN Acoustic Model #

This repository implements a manifold regularized acoustic model based on the relevant paper by Tomar and Rose, 2014.

Dependencies: Kaldi Speech Recognition Toolkit, Python + Theano library, Julia

The basic steps to run the code are outlined as follows:

* Having installed Kaldi, run src_dnn/run-GMM.sh to get the GMM/HMM that will create the initial alignments for the DNN.
* Run the feature-related parts of src_dnn/pfile_or_decode.sh (src_dnn/pfile_or_decode.sh "train") to get the training and validation pfiles for the DNN
* Configure and run src_dnn/train_dnn_NN.py for the baseline un-regularized DNN acoustic model: set manifold_data_spec, knn, gamma to None
* Install and run kaldi-trunk/src/bin/show-transitionStates to get the desired labels for the manifold graphs' construction, i.e. phones or pairs of phones-HMMstate. If you choose the second, run src_manifold/labelsPhonesHMMstates.m to get a mapping from unique pairs to integers 
* Configure and run src_manifold/getWint.jl,getWpen.jl,getNNneighborhoods_New.jl to get the affinity matrices Wint,Wpen as well as the manifold neighbors of each sample *in their context* (surrounding 8 frames) and the corresponding weights
* Run src_manifold/readbin_julia.jl and pipe to pfile_create (from 
pfile_utils_Float_Labels) to create the manifold pfile
* Configure and run src_dnn/train_dnn_NN.py for the manifold-regularized DNN acoustic model: set manifold_data_spec, knn, gamma to the desired values
* To evaluate in Kaldi, make sure you have provided a Kaldi model output file name in src_dnn/train_dnn_NN.py. Then, copy model in $working_dir/dnn.nnet and run decoding part of src_dnn/pfile_or_decode.sh (src_dnn/pfile_or_decode.sh "decode")
* If you need to use rectified linear units and decode in Kaldi, make sure you have installed in Kaldi the ReLU component found here: *https://github.com/naxingyu/kaldi-nn/tree/master/src/nnet*



Yannis Chalkiadakis, haljohn@hotmail.com

May 2016