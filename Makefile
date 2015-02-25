OBJS = kaldi2txt.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -O3 $(DEBUG)
LFLAGS = -Wall $(DEBUG)


kaldi2txt: $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o kaldi2txt


kaldi2txt.o: kaldi2txt.cpp
	$(CC) $(CFLAGS) -c kaldi2txt.cpp


clean:
	\rm *.o *~ kaldi2txt

