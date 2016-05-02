#!/bin/bash

#Train monophone and triphone GMM/HMM system on 2000 shortest utterances of WSJ

. ./cmd.sh 

#Set path to dataset files
wsj0=/media/data/ichalkia/wsj0
wsj1=/media/data/ichalkia/wsj1

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc;
for x in test_eval92 test_eval93 train_si284; do 
 steps/make_mfcc.sh --nj 1  data/$x exp/make_mfcc/$x $mfccdir;
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir 
done

sed 's/[-A-Za-z0-9]*\.wv1/\U&/' <data/test_dev93/wav.scp >data/test_dev93/wav.scp2;
mv data/test_dev93/wav.scp2 data/test_dev93/wav.scp
x=test_dev93 
steps/make_mfcc.sh --nj 1  data/$x exp/make_mfcc/$x $mfccdir;
steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir

utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

# Now make subset with the shortest 2k utterances from si-84.
utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
#utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;


steps/train_mono.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang_nosp exp/mono0a || exit 1;

(
 utils/mkgraph.sh --mono data/lang_nosp_test_tgpr \
    exp/mono0a exp/mono0a/graph_nosp_tgpr && \
 steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
    data/test_dev93 exp/mono0a/decode_nosp_tgpr_dev93 && \
 steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
    data/test_eval92 exp/mono0a/decode_nosp_tgpr_eval92 
) &

steps/align_si.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang_nosp exp/mono0a exp/mono0a_ali_2kshort || exit 1;

#lang_nosp : without position dependent 3phones
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \  
    data/train_si84_2kshort data/lang_nosp exp/mono0a_ali_2kshort exp/tri1_2kshort || exit 1;  

while [ ! -f data/lang_nosp_test_tgpr/tmp/LG.fst ] || \
   [ -z data/lang_nosp_test_tgpr/tmp/LG.fst ]; do
  sleep 20;
done
sleep 30;
# or the mono mkgraph.sh might be writing data/lang_test_tgpr/tmp/LG.fst which will cause this to fail.

utils/mkgraph.sh data/lang_nosp_test_tgpr \
  exp/tri1_2kshort exp/tri1_2kshort/graph_nosp_tgpr || exit 1;

steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/tri1_2kshort/graph_nosp_tgpr \
  data/test_dev93 exp/tri1_2kshort/decode_nosp_tgpr_dev93 || exit 1;
steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/tri1_2kshort/graph_nosp_tgpr \
  data/test_eval92 exp/tri1_2kshort/decode_nosp_tgpr_eval92 || exit 1;

steps/align_si.sh --nj 10 --cmd "$train_cmd" \   
  data/train_si84_2kshort data/lang_nosp exp/tri1_2kshort exp/tri1_2kshort_ali || exit 1;

# Train tri2a, which is deltas + delta-deltas, on 2kshort data.
steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 \
  data/train_si84_2kshort data/lang_nosp exp/tri1_2kshort_ali exp/triPhone_si84_2kshort || exit 1;

steps/align_si.sh --nj 8 --cmd "$train_cmd" \
  data/train_si84_2kshort data/lang_nosp exp/triPhone_si84_2kshort exp/triPhone_si84_2kshort_ali || exit 1;
 
utils/mkgraph.sh data/lang_nosp_test_tgpr exp/triPhone_si84_2kshort exp/triPhone_si84_2kshort/graph_nosp_tgpr || exit 1;

steps/decode.sh --nj 8 --cmd "$decode_cmd"  exp/triPhone_si84_2kshort/graph_nosp_tgpr  \
  data/test_dev93 exp/triPhone_si84_2kshort/decode_nosp_tgpr_dev93 || exit 1;
steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/triPhone_si84_2kshort/graph_nosp_tgpr  \
  data/test_eval92 exp/triPhone_si84_2kshort/decode_nosp_tgpr_eval92 || exit 1;


