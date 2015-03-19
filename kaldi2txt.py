#!/usr/bin/python

import sys
import getopt
import re

def main(argv):
   input_file = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:")
   except getopt.GetoptError:
      print 'No arguments passed'
      print 'kaldi2txt.py -i <inputfile> -o <outputfile>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print 'kaldi2txt.py -i <inputfile> -o <outputfile>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-o"):
          output_file = arg
   if input_file=='' or output_file=='':
       print 'kaldi2txt.py -i <inputfile> -o <outputfile>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       out_f = open(output_file,'w')

   cbracket = re.compile(r'\]')
   utt_obracket = re.compile(r'^\s*\S*\s*\[')
   empty = re.compile(r'^\s*$')
   for line in in_f:
      line1 = cbracket.sub('',line)
      line2 = utt_obracket.sub('',line1)
      match = empty.match(line2)
      if not match:
         out_f.write(line2)
     

   in_f.close()
   out_f.close()



if __name__ == "__main__":
   main(sys.argv[1:])
