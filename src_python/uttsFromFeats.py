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
      print 'uttsFromFeats.py -i <inputfile> -o <outputfile>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print 'uttsFromFeats.py -i <inputfile> -o <outputfile>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-o"):
          output_file = arg
   if input_file=='' or output_file=='':
       print 'uttsFromFeats.py -i <inputfile> -o <outputfile>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       out_f = open(output_file,'w')
   
   utt_obracket = re.compile(r'^\s*\S*\s*\/')
   for line in in_f:
      if utt_obracket.match(line):
         out_f.write(utt_obracket.match(line).group().replace("/"," ")+"\n")

         
   in_f.close()
   out_f.close()



if __name__ == "__main__":
   main(sys.argv[1:])
