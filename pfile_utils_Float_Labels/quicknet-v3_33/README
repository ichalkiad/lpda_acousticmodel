$Header: /u/drspeech/repos/quicknet2/README,v 1.11 2013/11/01 00:17:47 davidj Exp $

This is the README file for "QuickNet", a set of programs and
libraries that can be used to train MLPs (multi-layer perceptrons, a
type of "neural net").  QuickNet is focused on high-performance MLP
training and use for audio processing, although it may be applicable
to other domains.

For instructions on how to install the package, read the associated
`INSTALL' file.  For some preliminary information of use to those
using or modifying the sources, read the associated `README-hacker'
file (which is no doubt very out of date).

The library was developed by the Speech Group at the International
Computer Science Institute, Berkeley, California
(http://www.ICSI.Berkeley.EDU/Speech/).  The original coding was done
by David Johnson (http://www.icsi.berkeley.edu/~davidj) with guidance
and inspiration provided by Nelson Morgan.  Significant additions were
made by Dan Ellis, Chuck Wooters and Chris Oei.

Some advantages of QuickNet:
 - Very portable.  Pretty much all you need is a 32 bit C++ compiler and a
   standard C library.
 - Optionally very low resource requirements, even for training. 
    - You only need room in memory for one copy of the weights _even if
      the disk format is transposed from your MLP's format_.
    - You can do a forward pass when you don't even have enough memory to
      store one utterance in RAM
    - You can train from disk
    - Efficient with I/O - does a fair amount in one pass of the data
 - An MLP is a fundamental object, not layers or matrices.  This makes
   it easy to optimize code across layers and use weird hardware
   efficiently but makes it hard to change the MLP architecture.  
 - QuickNet is a library as well as an application
 - The code for constrained-latency (i.e. real time) forward pass is
   included (indeed it is just tweaks to the existing objects).  You can use
   exactly the same code for training as for a low-resource embedded systems.
 - It includes a useful I/O library and some signal processing functions.
 - It includes high performance CPU pthreads-based MLP code - no GPU required!
 - It iplements streams and segments (aka utterances) that map well to
   speech processing applications.

Some disadvantages:
 - Written in C++not a scripting language
    - A higher learning barrier for code modifications. 
    - Lots of the usual C++ overhead and verbosity, especially memory
      management and lots of work creating and tearing down objects.
 - Primitive C++ - doesn't even use STL.  This leads to some ugly
   macro stuff for streams and lots of messing around with char*.
 - Non-trivial overhead to write your own program using QuickNet as a
   library, especially given there's no proper enapsulation of the
   command line argument handling code.
 - Experimenting with network architectures is a lot of work.
 - The streams model is a bit broken - MLP objects should be streams.
 - Lots of the low resource and high performance code makes it hard to
   work on - ots of chasing pointers around buffers.
 - Some of the efficiency is probably less meaningful these days -
   gaining flexibility and losing a factor of 1.something in performance
   would seem to be a win.

Note that this is actually the README for QuickNet version 3 which was
branched from the original version of QuickNet, version 1, in early
2004.  See the `NEWS' file for the differences between QuickNet 1 and
QuickNet 3.

For more details on QuickNet, please see the QuickNet web page
(http://www.icsi.berkeley.edu/Speech/qn.html) or contact
<quicknet-info@ICSI.Berkeley.EDU>.  Be aware that QuickNet is not
supported in any official manner.  However, well-documented bug
reports that include logs and a way for data files to be accessed
(e.g. by FTP) may well get acted upon.


QuickNet is Copyright (c) 1997-2010 The International Computer Science
Institute.  All rights reserved.  See the file "COPYING" in the source
directory for more details.
