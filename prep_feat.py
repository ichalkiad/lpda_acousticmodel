#!/usr/bin/python

#./prep_feat.py  -i ../KALDI_norm_var/mfcc_norm_mean_var.txt -l ../KALDI_norm_var/missing_utter.txt -o mfcc_by_frame.txt -m cnt_fram.txt -k utter.txt

import sys
import getopt
import re

def main(argv):
   input_file = ''
   input_file2 = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:l:o:m:k:")
   except getopt.GetoptError:
      print 'No arguments passed'
      print 'kaldi2txt.py -i <inputfile1> -l <inputfile2>  -o <outputfile> -m <outputfile2> -k <outputfile3>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print 'kaldi2txt.py -i <inputfile1> -l <inputfile2> -o <outputfile> -m <outputfile2> -k <outputfile3>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-l"):
          input_file2 = arg
       elif opt in ("-o"):
          output_file = arg
       elif opt in ("-m"):
          output_file2 = arg
       elif opt in ("-k"):
          output_file3 = arg
   if input_file=='' or output_file=='':
       print 'kaldi2txt.py -i <inputfile> -l <inputfile2> -o <outputfile> -m <outputfile2> -k <outputfile3>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       label_f = open(input_file2,'r')
       out_f = open(output_file,'w')
       out_f2 = open(output_file2,'w')
       out_f3 = open(output_file3,'w')

   missingUtter=[] #if we want only the utts in label_f
   for line in label_f: 
       #missingUtter = line.split();
       missingUtter.append(line.split()[0])  #if we want only the utts in label_f
   
   cbracket = re.compile(r'\]')
   obracket = re.compile(r'^\s*\w*\s*\[')
   empty = re.compile(r'^\s*$')
   missing = 0;
   cnt_frames = 0
   for line in in_f:
      line1 = obracket.search(line)
      if line1:
         if (cnt_frames != 0):
            out_f2.write(repr(cnt_frames)+"\n")
            cnt_frames = 0
         missing =0 
         if line1.group().replace("[","").strip() in missingUtter:
            missing = 1
               
         #if obracket.match(line):
         #    print obracket.search(line).group().replace("["," ")
      
      if (missing == 1):  #if we want only the utts in label_f == 1, otherwise == 0
         if line1:
            out_f3.write(line1.group().replace("[","")+"\n")
         line2 = cbracket.sub('',line)
         line3 = obracket.sub('',line2)
         match = empty.match(line3)
         if not match:
            cnt_frames += 1
            out_f.write(line3)
   

   out_f2.write(repr(cnt_frames))
      
   in_f.close()
   out_f.close()
   label_f.close()
   out_f2.close()
   out_f3.close()


if __name__ == "__main__":
   main(sys.argv[1:])
