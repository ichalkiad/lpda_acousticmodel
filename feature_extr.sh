#!/bin/bash

JULIA=/usr/bin/julia
KALDI=/home/yannis/kaldi-trunk
INPUT_FDR=/home/yannis/Desktop/mfcc
OUTPUT_FDR=/home/yannis/Desktop/LPDA

#Convert .ark to text files

$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.1.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_1_txt.ark

$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.2.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_2_txt.ark

#Remove sample delimiters for Julia processing

./kaldi2txt raw_mfcc_train_si284_1_txt.ark raw_mfcc_train_si284_1_txt_formatted.ark
./kaldi2txt raw_mfcc_train_si284_2_txt.ark raw_mfcc_train_si284_2_txt_formatted.ark

#Make HDF5 file for Julia/MatLab processing

$JULIA txt2julia.jl OUTPUT_FDR/raw_mfcc_train_si284_1_txt_formatted.ark 

$JULIA txt2julia.jl OUTPUT_FDR/raw_mfcc_train_si284_2_txt_formatted.ark
