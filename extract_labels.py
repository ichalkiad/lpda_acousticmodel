#!/usr/bin/python

import sys
import getopt

def main(argv):
   input_file = ''
   output_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:")
   except getopt.GetoptError:
      print 'No arguments passed'
      print 'extract_labels.py -i <inputfile> -o <outputfile>'
      sys.exit()
   for opt, arg in opts:
       if opt == '-h':
          print 'extract_labels.py -i <inputfile> -o <outputfile>'
          sys.exit()
       elif opt in ("-i"):
          input_file = arg
       elif opt in ("-o"):
          output_file = arg
   if input_file=='' or output_file=='':
       print 'extract_labels.py -i <inputfile> -o <outputfile>'
       sys.exit()
   else:
       in_f = open(input_file,'r')
       out_f = open(output_file,'w')

   trans ='Transition-state'
   for line in in_f:
      if (line.find(trans)!=-1):
         tokens = line.split()
         phone = tokens[4]
         hmm_state = tokens[7]
         out_f.write(phone+'-'+hmm_state+'\n')
         

   in_f.close()
   out_f.close()



if __name__ == "__main__":
   main(sys.argv[1:])
