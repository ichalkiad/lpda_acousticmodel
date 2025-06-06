$Header: /u/drspeech/repos/quicknet2/README-hacker,v 1.11 2013/11/01 00:18:34 davidj Exp $

Some Notes on the Source of QuickNet
====================================

Introduction
------------

This file contains some notes that might be of use if you are either
using QuickNet as a library or modifying it in some way.  Note that the
details here may not be completely correct as many folks have changed
QuickNet and few have updated documentation.


Rationale
---------

Some of the design goals guiding the implementation of QuickNet are
described below, along with some of the consequences of these
considerations.

i) High performance

Most classes pass around multiple frames/presentations/features,
rather than just one.  This reduces the impact of `overhead' such as
virtual function dispatch.  It also allows vector or matrix routines
to be used in implementations many places.

ii) Useful as a library

Some care has been put into the diagnostic features of classes to aid
in use as a library.  Specifically, different levels of logging
output for each class aids debugging.

iii) Independent of file formats

All file access routines are based on abstract classes



Current Status
--------------

The example programs work and are in regular use at ICSI.
Functionality of the various classes used by these programs is
probably safe to use.

As a crude guide, classes and routines beginning `QN_' are much more
likely to be closer to their final form than those without this
prefix.  This is no indication of their correctness, but if you use
them in the way they are written now, there will be a lower
probability of your program needing to be changed in future.


The Components of QuickNet
--------------------------

The components of the QuickNet package can be grouped into several
distinct parts.

i) A set of classes implementing MLPs

The MLP classes are based on the abstract class `QN_MLP', with the
hope that you will be able to change your MLP implementation without
changing your whole program.  The `QN_MLP' class is designed to hide
as much of the MLP implementation as possible, hence there is no
direct access to the weight matrix data structures (although there are
member functions that access them).  Also, although the current
implementations are for 3 layer MLPs, this is not inherent in the
interface.

The current ipmlementations of MLPs are:

QN_MLP_OnlineFl3
	An online, floating point, fully connected, 3 layer MLP.
QN_MLP_BunchFl3
	A "bunch" (aka batch) mode, floating point, fully connected
	3 layer MLP.  The size of batches is variable - a batch size
        of one results in a net that trains like QN_MLP_OnlineFl3.
QN_MLP_ThreadFl3
	A "bunch" (aka batch) mode, floating point, fully connected
	3 layer MLP that uses Posix threads to exploit parallelism on
        multiprocessor machines. 
QN_MLP_BunchFlVar
	A "bunch" (aka batch) mode, floating point, fully connected
	MLP that can have 3, 4 or 5 layers.
QN_MLP_FlVar
	A "bunch" (aka batch) mode, floating point, fully connected
	MLP that can have 3, 4 or 5 layers and that uses Posix threads
	to exploit parallelisom on a multiprocessor machine.

ii) A set of classes abstracting access to various file formats

iii) Utility routines and classes

Basically, code for doing things with MLPs and files.

The `QN_write_weights' and `QN_read_weights' functions are finished and
working.


iv) Example programs

There are currently four simple programs using the above classes:

     qnmultitrn - train 3, 4 or 5 layer MLPs
     qnmultifwd - forward pass for 3, 4 or 5 layer MLPs
     qnnorm - simple normalization of feature values
     qncopy - copy/transform a feature file (limit funcitonality - check out
              feacat and pfile_utils packages for a wider range of features)
     qncopywts - copy/transform a weights file
     qnxor - a trivial MLP example
     qnstrn - train 3 layer MLPs (older - deprecated)
     qnsfwd - forward pass for 3 layer MLPs (older - deprecated)


Using the testsuite
-------------------

A simple library test suite included with the QuickNet distribution.
The aim of the test suite is to test the library functions, as opposed
to testing the complete applications.  Currently the test suite is by
no means comprehensive, but it is a useful check that your build has
completed successfully.  To use the testsuite, you will need to have
the `rtst' library installed, along with the latest version of the
`quicknet_testdata' package.

To run the testsuite, move to the `testsuite' subdirectory of the
build directory and type `gnumake check'.  This will build the test
programs and then run them.  Running the tests should only take a few
minutes.  Successful completion is indicated by the lack of error
messages and a clean return from the make.  The testsuite should work
on both SPERT and workstations.


Source Code Repository
----------------------

The CVS source code repository for QuickNet2 is in
/u/drdspeech/repos/quicknet2/ at ICSI.  This is only accessible to
people with accounts at ICSI.


Releasing QuickNet
------------------

Before installing at ICSI or distributing offsite a new version should
be released.  Numeric versions, e.g.  v3_21, are assumed to be stable.
Unstable or prerelease versions should have a suffix, e.g. v3_21pre or
v3_21djhack.

The process of releasing consists of:

 - Update NEWS
 - Check documentation - INSTALL, README, README-hacker, man pages
 - Update version in configure.in
 - Do a "make" so autoconf builds configure
 - Check everything into CVS, including "cvs commit -f stamp-h.in"
 - Add a tag to everything in CVS

	cd $(srcdir)
	cvs tag -c v3_99

 - Make the distribution tar

	cd ../DIST
	gmake dist

 - If a stable release, copy the distribution tar to the FTP directory
   /u/ftp/pub/real/davidj/ and change the link in that directory to
   point to the new version.

