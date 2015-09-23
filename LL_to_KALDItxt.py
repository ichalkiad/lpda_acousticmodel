#!/usr/bin/python

# sed 's/\[//g;s/\]//g' /path/to/file >> clean_file
#./LL_to_KALDItxt.py  -i utter.txt -l cnt_fram.txt  -m mfcc_by_frame.txt -o test.txt
#./copy-matrix ark,t:/home/yannis/Desktop/new_feats--KALDI_setup/test.txt ark,scp:test.ark,test.scp   


import sys
import getopt
import re
import numpy as np

def main(argv):
   input_file = ''
   input_file2 = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:l:m:o:")
   except getopt.GetoptError:
      print 'No arguments passed'
      print 'LL_to_KALDItxt.py -i <utterance-ids> -l <#frames_in_each_utterance>  -m <predictions_file>  -o <ll_per_frame_indexed_by_utt>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print  'LL_to_KALDItxt.py -i <utterance-ids> -l <#frames_in_each_utterance> -m <predictions_file>  -o <ll_per_frame_indexed_by_utt>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-l"):
          input_file2 = arg
       elif opt in ("-m"):
          input_file3 = arg
       elif opt in ("-o"):
          output_file = arg
       
   if input_file=='' or output_file=='':
       print  'LL_to_KALDItxt.py -i <utterance-ids> -l <#frames_in_each_utterance> -m <predictions_file> -o <ll_per_frame_indexed_by_utt>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       pred_f = open(input_file3,'r')
       label_f = open(input_file2,'r')
       out_f = open(output_file,'w')
       
   utterances = np.array(in_f.read().split());
   num_of_frames = np.array(label_f.read().split());
   a = num_of_frames.astype(int)
   
   for i in xrange(len(utterances)):
       for j in xrange(a[i]):
         line = next(pred_f)
         if (j==0):
            out_f.write(utterances[i]+" "+"[ "+line)
         elif (j==a[i]-1):
            out_f.write(line.replace("\n"," ]")+"\n")
         else:
            out_f.write(line)
#       print "i=" + repr(i)
#       print a[i]


   in_f.close()
   pred_f.close()
   out_f.close()
   label_f.close()


if __name__ == "__main__":
   main(sys.argv[1:])
