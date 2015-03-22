#!/bin/bash

JULIA=/usr/bin/julia
KALDI=/home/yannis/kaldi-trunk
INPUT_FDR=/home/yannis/Desktop/mfcc
OUTPUT_FDR=/home/yannis/Desktop/LPDA

#Convert .ark to text files

$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.1.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_1_txt.ark

$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.2.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_2_txt.ark

echo 'Done converting .ark to .txt.'

#Remove sample delimiters for Julia processing; first set python path in kaldi2txt.py

./kaldi2txt.py -i raw_mfcc_train_si284_1_txt.ark -o raw_mfcc_train_si284_1_txt_formatted.ark
./kaldi2txt.py -i raw_mfcc_train_si284_2_txt.ark -o raw_mfcc_train_si284_2_txt_formatted.ark

echo 'Removed unnecessary delimiters.'

#Make binary file for Julia/MatLab processing

$JULIA txt2julia.jl raw_mfcc_train_si284_1_txt_formatted.ark $OUTPUT_FDR $OUTPUT_FDR 

$JULIA txt2julia.jl raw_mfcc_train_si284_2_txt_formatted.ark $OUTPUT_FDR $OUTPUT_FDR

echo 'Feature file ready for processing.'
