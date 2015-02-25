#include<iostream>
#include<fstream>
#include<string>
#include<sys/stat.h>
#include<sys/types.h>
#include<stdlib.h>

using namespace std;

int main(int argc, char *argv[])
{
  int p,skip = 0;
   const char* buf = "feat.txt";
   ifstream input_file(argv[1]);
   char line[256];
   string line_s;
   ofstream out_file(buf);
   //read the input file line by line
   if(input_file.is_open()) {
      while (input_file.get()!=-1) {
       input_file.getline(line,256);
       line_s = string(line);
       //skip the header of each sample
       p = line_s.find_first_of("[");
       if (p != string::npos) { 
	 skip = 1;
       }
       //avoid the end mark of each sample
       p = line_s.find_first_of("]");
       if (p != string::npos) { 
           line_s.erase(p);
       }
       if (out_file.is_open()) {
	   if (!skip)
	     out_file << line_s << endl;
       }
       else {
	 cout << "Error opening output file." << endl;
	 exit(1);
       }
       skip = 0;
      }
   }
   else {
     cout << "Error opening input file." << endl;
     exit(1);
   }
   out_file.close();
   input_file.close();
   
   return 0;
   
}
