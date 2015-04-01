#!/bin/bash

JULIA=/usr/bin/julia
KALDI=/home/yannis/kaldi-trunk
INPUT_FDR=/home/yannis/Desktop/mfcc
OUTPUT_FDR=/home/yannis/Desktop/LPDA

#Convert .ark to text files

#$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.1.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_1_txt.ark

#$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.2.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_2_txt.ark

echo 'Done converting .ark to .txt.'

#Remove features of unaligned utterances

#./uttsFromFeats.py -i feats.scp -o total_utterances.txt
#diff -b -u labelled_utterances_total.txt total_utterances.txt | grep -E  "^\+" >> missingUtterances.txt
#./getUnalignedUtts.py -i missingUtterances.txt -o missing_utter.txt
#rm total_utterances.txt missingUtterances.txt

echo 'Done clearing  up feature list.'

#Remove sample delimiters for Julia processing; first set python path in kaldi2txt.py

#./kaldi2txt.py -i raw_mfcc_train_si284_1_txt.ark -l missing_utter.txt -o mfcc_si284_1.txt
#./kaldi2txt.py -i raw_mfcc_train_si284_2_txt.ark -l missing_utter.txt -o mfcc_si284_2.txt

echo 'Done removing unnecessary delimiters.'

#Make binary file for Julia/MatLab processing

$JULIA txt2julia_ener.jl mfcc_si284_1.txt $OUTPUT_FDR $OUTPUT_FDR 

$JULIA txt2julia_ener.jl mfcc_si284_2.txt $OUTPUT_FDR $OUTPUT_FDR

echo 'Features file ready for processing.'

#$JULIA txt2julia_labels.jl labelled_utterances_total.txt $OUTPUT_FDR $OUTPUT_FDR

echo 'Labels file ready for processing.'
