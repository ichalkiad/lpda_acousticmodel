#!/bin/sh
#
# $Header: /u/drspeech/repos/quicknet2/testdata_qnnorm.sh.in,v 1.9 2004/03/26 06:00:43 davidj Exp $
#
# testdata_qnnorm.sh.  Generated from testdata_qnnorm.sh.in by configure.
#
# This script runs the `qnnorm' program, but sets up a useful set of 
# default parameters that use the data files in the `testdata' subdirectory.
# The specificied parameters can be over ridden by using
# "param=value" on the command line.

# This bit takes command line arguments of the form "var=val", and sets
# sh varible `var' to `val'.

while [ $# -gt 0 ]; do
        case "$1" in
        *=*)    key=`echo "$1" | sed "s/=.*//"`
                val=`echo "$1" | sed "s/[^=]*=//"`
		if [ "$val" ]; then
	                eval "$key"=\'"$val"\'
		fi
                unset key val
                shift ;;
        *)      break;;
        esac
done

# The lines below will have substitution made in the configure process
: ${testdata_dir:=/media/data/ichalkia/kaldi-trunk/tools/quicknet-v3_33/share/quicknet_testdata}

${qnnorm:=./qnnorm} \
	norm_ftrfile=${norm_ftrfile:=$testdata_dir/train1-rasta8+d.pfile} \
	output_normfile=${output_normfile:=testdata.norm} \
	first_sent=${first_sent:=0} \
	num_sents=${num_sents:=1720} \
	debug=${debug:=0} \
	verbose=${verbose:=true}

