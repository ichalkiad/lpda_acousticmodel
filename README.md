# Manifold-Regularized DNN Acoustic Model #

This repository implements a manifold regularized acoustic model based on the relevant paper by Tomar and Rose, 2014.

Dependencies: Kaldi Speech Recognition Toolkit, Python + Theano library, Julia

The basic steps to run the code are outlined as follows:

* Having installed Kaldi, run src_dnn/run-GMM.sh to get the GMM/HMM that will create the initial alignments for the DNN.
* Run the feature-related parts of src_dnn/pfile_or_decode.sh to get the training and validation pfiles for the DNN
* Configure and run src_dnn/train_dnn_NN.py for the baseline un-regularized DNN acoustic model: set manifold_data_spec, knn, gamma to None
* Install and run kaldi-trunk/src/bin/show-transitionStates to get the desired labels for the manifold graphs' construction, i.e. phones or pairs of phones-HMMstate. If you choose the second, run src_manifold/labelsPhonesHMMstates.m to get a mapping from unique pairs to integers 
* Configure and run src_manifold/getWint.jl,getWpen.jl,getNNneighborhoods_New.jl to get the affinity matrices Wint,Wpen as well as the manifold neighbors of each sample *in their context* (surrounding 8 frames) and the corresponding weights
* Run src_manifold/readbin_julia.jl and pipe to pfile_create (from 
pfile_utils_Float_Labels) to create the manifold pfile








Yannis Chalkiadakis, haljohn@hotmail.com

May 2016