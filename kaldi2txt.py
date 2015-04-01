#!/usr/bin/python

import sys
import getopt
import re

def main(argv):
   input_file = ''
   input_file2 = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:l:o:")
   except getopt.GetoptError:
      print 'No arguments passed'
      print 'kaldi2txt.py -i <inputfile1> -l <inputfile2>  -o <outputfile>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print 'kaldi2txt.py -i <inputfile1> -l <inputfile2> -o <outputfile>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-l"):
          input_file2 = arg
       elif opt in ("-o"):
          output_file = arg
   if input_file=='' or output_file=='':
       print 'kaldi2txt.py -i <inputfile> -l <inputfile2> -o <outputfile>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       label_f = open(input_file2,'r')
       out_f = open(output_file,'w')
       
   for line in label_f: 
       missingUtter = line.split();

   cbracket = re.compile(r'\]')
   obracket = re.compile(r'^\s*\w*\s*\[')
   empty = re.compile(r'^\s*$')
   missing = 0;
   for line in in_f:
      line1 = obracket.search(line)
      if line1:
         missing = 0 
         if line1.group().replace("[","").strip() in missingUtter:
            missing = 1
         
      #if obracket.match(line):
      #    print obracket.search(line).group().replace("["," ")
       
      if (missing == 0):
         #if line1:
         #   print line1.group().replace("[","")
         line2 = cbracket.sub('',line)
         line3 = obracket.sub('',line2)
         match = empty.match(line3)
         if not match:
            out_f.write(line3)
         
   in_f.close()
   out_f.close()
   label_f.close()


if __name__ == "__main__":
   main(sys.argv[1:])
