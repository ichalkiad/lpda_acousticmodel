#!/bin/bash

MATLAB=/usr/local/bin/matlab
JULIA=/usr/bin/julia
KALDI=/home/yannis/kaldi-trunk
INPUT_FDR=/home/yannis/Desktop/mfcc
OUTPUT_FDR=/home/yannis/Desktop/LPDA

#JULIA=julia 
#KALDI=/media/data/ichalkia/kaldi-trunk
#INPUT_FDR=/media/data/ichalkia/kaldi-trunk/egs/wsj/s5/mfcc
#OUTPUT_FDR=/home/ichalkia

#Convert .ark to text files

#$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/raw_mfcc_train_si284.1.ark ark,t:$OUTPUT_FDR/raw_mfcc_train_si284_txt.ark


echo 'Done converting .ark to .txt.'

#Remove features of unaligned utterances

#./uttsFromFeats.py -i $KALDI/egs/wsj/s5/data/train_si84_half/feats.scp -o total_utterances.txt
#diff -b -u /home/ichalkia/andromeda_scripts/labelled_utterances.txt total_utterances.txt | grep -E  "^\+" >> missingUtterances.txt
#./getUnalignedUtts.py -i missingUtterances.txt -o $OUTPUT_FDR/missing_utter.txt
#rm total_utterances.txt missingUtterances.txt


#./uttsFromFeats.py -i $KALDI/egs/wsj/s5/data/train_si284/feats.scp -o total_utterances_si284.txt
#diff -b -u total_utterances.txt total_utterances_si284.txt | grep -E  "^\+" >> missingUtterances.txt
#./getUnalignedUtts.py -i missingUtterances.txt -o $OUTPUT_FDR/missing_utter_feat.txt
#rm total_utterances_si284.txt missingUtterances.txt



echo 'Done clearing  up feature list.'

#Remove sample delimiters for Julia processing; first set python path in kaldi2txt.py

#./kaldi2txt.py -i $OUTPUT_FDR/raw_mfcc_train_si284_txt.ark -l $OUTPUT_FDR/missing_utter_feat.txt -o $OUTPUT_FDR/mfcc_si84_half.txt
#./kaldi2txt.py -i $OUTPUT_FDR/mfcc_si84_half.txt -l $OUTPUT_FDR/missing_utter.txt -o $OUTPUT_FDR/mfcc_si84_half_clean.txt


echo 'Done removing unnecessary delimiters.'

#Make binary file for Julia/MatLab processing

#$JULIA -p 12 $OUTPUT_FDR/andromeda_scripts/txt2julia_ener.jl mfcc_si84_half_clean.txt $OUTPUT_FDR $OUTPUT_FDR
$JULIA -p 2 txt2julia_ener.jl mfcc_si84_half_clean.txt $OUTPUT_FDR $OUTPUT_FDR 

echo 'Features file ready for processing.'

#$JULIA -p 12 txt2julia_labels.jl frameLabels.txt $OUTPUT_FDR/andromeda_scripts $OUTPUT_FDR

echo 'Labels file ready for processing.'

#$MATLAB -nodisplay  < /home/yannis/Desktop/LPDA/getPhonesPriors.m

echo 'Labels priors extracted.'
