#!/bin/bash

MATLAB=/usr/local/bin/matlab
JULIA=/usr/bin/julia
KALDI=/home/yannis/kaldi-trunk
INPUT_FDR=/home/yannis/Desktop/KALDI_norm_var
OUTPUT_FDR=/home/yannis/Desktop/KALDI_norm_var


#Convert .ark to text files

#$KALDI/src/featbin/copy-feats ark:$INPUT_FDR/mfcc_norm_mean_var.ark ark,t:$OUTPUT_FDR/mfcc_norm_mean_var.txt


echo 'Done converting .ark to .txt.'

#Remove features of unaligned utterances

./uttsFromFeats.py -i $INPUT_FDR/feats_si284.scp -o total_utterances.txt
diff -b -u $INPUT_FDR/labelled_utterances_all.txt total_utterances.txt | grep -E  "^\+" >> missingUtterances.txt
./getUnalignedUtts.py -i missingUtterances.txt -o $OUTPUT_FDR/missing_utter.txt
rm total_utterances.txt missingUtterances.txt


echo 'Done clearing  up feature list.'

#Remove sample delimiters for Julia processing; first set python path in kaldi2txt.py

./kaldi2txt.py -i $INPUT_FDR/mfcc_norm_mean_var.txt -l $OUTPUT_FDR/missing_utter.txt -o $OUTPUT_FDR/mfcc_normMV_nodelim.txt
#./kaldi2txt.py -i $OUTPUT_FDR/mfcc_si84_half.txt -l $OUTPUT_FDR/missing_utter.txt -o $OUTPUT_FDR/mfcc_si84_half_clean.txt


echo 'Done removing unnecessary delimiters.'

#Make binary file for Julia/MatLab processing

$JULIA txt2julia.jl mfcc_normMV_nodelim.txt $INPUT_FDR $OUTPUT_FDR

echo 'Features file ready for processing.'

$JULIA txt2julia_labels.jl frameLabels2a.txt $INPUT_FDR $OUTPUT_FDR

echo 'Labels file ready for processing.'

$MATLAB -nodisplay  < /home/yannis/Desktop/LPDA/getPhonesPriors.m

echo 'Labels priors extracted.'



##### Andromeda

#JULIA=julia
#KALDI=/media/data/ichalkia/kaldi-trunk
#INPUT_FDR=/media/data/ichalkia/kaldi-trunk/egs/wsj/s5/mfcc
#OUTPUT_FDR=/home/ichalkia

#./uttsFromFeats.py -i $KALDI/egs/wsj/s5/data/train_si284/feats.scp -o total_utterances_si284.txt
#diff -b -u total_utterances.txt total_utterances_si284.txt | grep -E  "^\+" >> missingUtterances.txt
#./getUnalignedUtts.py -i missingUtterances.txt -o $OUTPUT_FDR/missing_utter_feat.txt
#rm total_utterances_si284.txt missingUtterances.txt

#$JULIA -p 12 $OUTPUT_FDR/andromeda_scripts/txt2julia_ener.jl mfcc_si84_half_clean.txt $OUTPUT_FDR $OUTPUT_FDR
