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
      print 'getUnalignedUtts.py -i <inputfile1> -o <outputfile>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print 'getUnalignedUtts.py -i <inputfile1> -o <outputfile>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-o"):
          output_file = arg
   if input_file=='' or output_file=='':
       print 'getUnalignedUtts.py -i <inputfile1> -o <outputfile>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       out_f = open(output_file,'w')
       
   utterance =  re.compile(r'^\s*\S*')
   
   for line in in_f:
      if utterance.match(line):
         line1 = utterance.search(line)
         out_f.write(line1.group().replace("+"," ")+" ")
              
   in_f.close()
   out_f.close()
   


if __name__ == "__main__":
   main(sys.argv[1:])
