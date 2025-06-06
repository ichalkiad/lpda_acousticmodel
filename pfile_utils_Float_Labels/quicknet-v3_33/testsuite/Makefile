# $Header: /u/drspeech/repos/quicknet2/testsuite/Makefile.in,v 1.62 2013/10/29 04:16:43 davidj Exp $
#
# Makefile.in for quicknet testsuite
# davidj - Wed Jun  7 11:19:10 1995


# Standard things we need

.SUFFIXES: 
.SUFFIXES: .c .cc .o .exe .run .s .dis

SHELL=/bin/sh

# Details of our host and target

build_alias = 
host_alias = 

srcdir = .
top_srcdir = ..

testdata_dir = /media/data/ichalkia/kaldi-trunk/tools/quicknet-v3_33/share/quicknet_testdata

# Installation stuff

prefix = /media/data/ichalkia/kaldi-trunk/tools/quicknet-v3_33
exec_prefix = ${prefix}
libdir = $(exec_prefix)/lib
includedir = $(exec_prefix)/include

# Programs we use

CC = gcc
CXX = g++
LD = $(CXX)
AR = @AR@
RANLIB = ranlib
INSTALL = /usr/bin/install -c
INSTALL_DATA = ${INSTALL} -m 644

# Tool arguments

CPPFLAGS = 
CFLAGS = -g -O2
CXXFLAGS = -g -O2
SHLIB_CFLAGS = -fPIC
SHLIB_CXXFLAGS = -fPIC
RTST_CFLAGS = 
RTST_LIBS = 
LDFLAGS = 
TESTFLAGS =
DEFS = -DHAVE_CONFIG_H
LIBS = -lpthread -lm 

libdir = ..
incs =  -I$(srcdir) -I$(top_srcdir) -I$(libdir) -I$(exec_prefix)/include -I$(prefix)/include
cflags =  $(DEFS) $(incs) $(CFLAGS) $(CPPFLAGS) $(SHLIB_CFLAGS) $(RTST_CFLAGS)
cxxflags = $(DEFS) $(incs) $(CXXFLAGS) $(CPPFLAGS) $(SHLIB_CXXFLAGS) $(RTST_CFLAGS)
ldflags = -L.. $(LDFLAGS)
testflags = $(TESTFLAGS)
libs = -L$(libdir) -lquicknet3 $(RTST_LIBS) $(LIBS) 
compile.c = $(CC) $(cflags)
compile.cc = $(CXX) $(cxxflags)
link.o = $(LD) $(ldflags)
quicknetlib = $(libdir)/libquicknet3.a

################################################################
# Implicit rules
################################################################

.c.o:
	$(compile.c) -c $<

.cc.o:
	$(compile.cc) -c $<

.o.exe:
	$(link.o) $< -o $@ $(libs)

.exe.run:
	./$< $(testflags)

################################################################
# The default target
################################################################

default: all


################################################################
# Variables we accumulate into
################################################################

all_srcs =
all_objs =
all_hdrs =
all_libs =
all_progs =
all_tests = 
garbage =

######################################################################
# The library tests
######################################################################

### Test the endianness routines.

all_srcs += endtest.cc
all_objs += endtest.o
all_progs += endtest.exe
all_tests += endtest.run
garbage += tmp.data

endtest.run: endtest.exe
	./endtest.exe $(testflags) tmp.data

### Test the MLP weight file classes ###

all_srcs += MLPWeightFile_test.cc
all_objs += MLPWeightFile_test.o
all_progs += MLPWeightFile_test.exe
all_tests += MLPWeightFile_test.run
garbage += tmp.weights

MLPWeightFile_test.run: MLPWeightFile_test.exe
	./MLPWeightFile_test.exe $(testflags) \
		$(testdata_dir)/ntimit-rasta-phones-200.weights \
		tmp.weights 153 200 56

all_srcs += MLPWeightFile_test2.cc
all_objs += MLPWeightFile_test2.o
all_progs += MLPWeightFile_test2.exe
all_tests += MLPWeightFile_test2.run
garbage += tmp2.weights

MLPWeightFile_test2.run: MLPWeightFile_test2.exe
	./MLPWeightFile_test2.exe $(testflags) \
		tmp2.weights 3 153 200 56

### Test PFile handling ###

all_srcs += PFile_test1.cc
all_objs += PFile_test1.o
all_progs += PFile_test1.exe
all_tests += PFile_test1.run

PFile_test1.run: PFile_test1.exe $(testdata_dir)/small.pfile \
				$(testdata_dir)/small-noindex.pfile
	./PFile_test1.exe $(testflags) \
		$(testdata_dir)/small.pfile 38 1963 18 1
	./PFile_test1.exe $(testflags) \
		$(testdata_dir)/small-noindex.pfile 38 1963 18 1
	./PFile_test1.exe $(testflags) \
		$(testdata_dir)/small-altformat.pfile 38 1963 18 1

# A test for writing PFiles.

all_srcs += PFile_test2.cc
all_objs += PFile_test2.o
all_progs += PFile_test2.exe
all_tests += PFile_test2.run
garbage += temp.pfile

PFile_test2.run: PFile_test2.exe
	./PFile_test2.exe $(testflags) temp.pfile 0
	./PFile_test2.exe $(testflags) temp.pfile 1


# A test for writing PFiles that contain just labels.

all_srcs += PFile_test3.cc
all_objs += PFile_test3.o
all_progs += PFile_test3.exe
all_tests += PFile_test3.run
garbage += temp.pfile

PFile_test3.run: PFile_test3.exe
	./PFile_test3.exe $(testflags) temp.pfile

### SRI Cambridge format files ###

all_srcs += sri_test1.cc
all_objs += sri_test1.o
all_progs += sri_test1.exe
all_tests += sri_test1.run

sri_test1.run: sri_test1.exe
	./sri_test1.exe $(testflags) $(testdata_dir)/train1-rasta8+d-first3.list $(testdata_dir) 3 18


### Test Cambridge format files ###

all_srcs += camfiles_test1.cc
all_objs += camfiles_test1.o
all_progs += camfiles_test1.exe
all_tests += camfiles_test1.run

camfiles_test1.run: camfiles_test1_pre_ind.run camfiles_test1_pre_noind.run \
		camfiles_test1_onl_ind.run camfiles_test1_onl_noind.run \
		camfiles_test1_lna8_ind.run camfiles_test1_lna8_noind.run
	@true			# Dummy command to override default rule

# Test pre file with an index
camfiles_test1_pre_ind.run: camfiles_test1.exe
	./camfiles_test1.exe $(testflags) \
		$(testdata_dir)/example.p16 p 20 3958 13 1 1

# Test pre file without an index
camfiles_test1_pre_noind.run: camfiles_test1.exe
	./camfiles_test1.exe $(testflags) \
		$(testdata_dir)/example.p16 p 20 3958 13 1 0

# Test online feature file with an index
camfiles_test1_onl_ind.run: camfiles_test1.exe
	./camfiles_test1.exe $(testflags) \
		$(testdata_dir)/fordavid.onlftr o 13 3531 9 0 1

# Test online feature file without an index
camfiles_test1_onl_noind.run: camfiles_test1.exe
	./camfiles_test1.exe $(testflags) \
		$(testdata_dir)/fordavid.onlftr o 13 3531 9 0 0

# Test LNA8 file with an index
camfiles_test1_lna8_ind.run: camfiles_test1.exe
	./camfiles_test1.exe $(testflags) \
		$(testdata_dir)/foo.lna l 650 24052 3 0 1

# Test LNA8 file without an index
camfiles_test1_lna8_noind.run: camfiles_test1.exe
	./camfiles_test1.exe $(testflags) \
		$(testdata_dir)/foo.lna l 650 24052 3 0 0

all_srcs +=  LNA8_test2.cc onlftr_test2.cc
all_objs +=  LNA8_test2.o onlftr_test2.o
all_progs +=  LNA8_test2.exe onlftr_test2.exe
all_tests += LNA8_test2.run onlftr_test2.run
garbage += temp.lna temp.onlftr

# Test reading then writing lna files
LNA8_test2.run: LNA8_test2.exe
	./LNA8_test2.exe $(testflags) temp.lna

# Test reading then writing onlftr files
onlftr_test2.run: onlftr_test2.exe
	./onlftr_test2.exe $(testflags) temp.onlftr

### Test some filters

all_srcs += InFtrStream_Narrow_test.cc InFtrStream_Norm_test.cc
all_objs += InFtrStream_Narrow_test.o InFtrStream_Norm_test.o
all_progs += InFtrStream_Narrow_test.exe InFtrStream_Norm_test.exe
all_tests += InFtrStream_Narrow_test.run InFtrStream_Norm_test.run

InFtrStream_Narrow_test.run: InFtrStream_Narrow_test.run1 \
				InFtrStream_Narrow_test.run2
	@true

# Small buffer.

InFtrStream_Narrow_test.run1: InFtrStream_Narrow_test.exe
	./InFtrStream_Narrow_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 1 10 1000

# Big buffer, just last feature.

InFtrStream_Narrow_test.run2: InFtrStream_Narrow_test.exe
	./InFtrStream_Narrow_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 17 a 50000

InFtrStream_Norm_test.run: InFtrStream_Norm_test.exe
	./InFtrStream_Norm_test.exe $(testflags) \
		$(testdata_dir)/small.pfile

### Test "stream surgery" functions.

all_srcs += cut_test.cc
all_objs += InFtrStream_Cut_test.o
all_progs += InFtrStream_Cut_test.exe
all_tests += InFtrStream_Cut_test.run

InFtrStream_Cut_test.o: $(srcdir)/cut_test.cc
	$(compile.cc) -c $(srcdir)/cut_test.cc -o InFtrStream_Cut_test.o

InFtrStream_Cut_test.run: InFtrStream_Cut_test.exe
	./InFtrStream_Cut_test.exe $(testflags) \
		$(testdata_dir)/small.pfile

all_srcs += InFtrStream_SplitFtrLab_test.cc
all_objs += InFtrStream_SplitFtrLab_test.o
all_progs += InFtrStream_SplitFtrLab_test.exe
all_tests += InFtrStream_SplitFtrLab_test.run

InFtrStream_SplitFtrLab_test.run: InFtrStream_SplitFtrLab_test.run1 \
					InFtrStream_SplitFtrLab_test.run2 \
					InFtrStream_SplitFtrLab_test.run3
	@true

InFtrStream_SplitFtrLab_test.run1: InFtrStream_SplitFtrLab_test.exe
	./InFtrStream_SplitFtrLab_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 10

InFtrStream_SplitFtrLab_test.run2: InFtrStream_SplitFtrLab_test.exe
	./InFtrStream_SplitFtrLab_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 100

InFtrStream_SplitFtrLab_test.run3: InFtrStream_SplitFtrLab_test.exe
	./InFtrStream_SplitFtrLab_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 1000

## Converting a label stream into a feature stream.

all_srcs += InFtrStream_OneHot_test.cc
all_objs += InFtrStream_OneHot_test.o
all_progs += InFtrStream_OneHot_test.exe
all_tests += InFtrStream_OneHot_test.run

InFtrStream_OneHot_test.run: InFtrStream_OneHot_test.run1
	@true

InFtrStream_OneHot_test.run1: InFtrStream_OneHot_test.exe
	./InFtrStream_OneHot_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 56 10

### Test some sequential windowing functions.

all_srcs += SeqWindow_test.cc
all_objs += SeqWindow_test.o
all_progs += SeqWindow_test.exe
all_tests += SeqWindow_test.run

SeqWindow_test.run: SeqWindow_test.run1 SeqWindow_test.run2 SeqWindow_test.run3
	@true

# No margins.

SeqWindow_test.run1: SeqWindow_test.exe
	./SeqWindow_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 9 0 0 d

# One frame window with margins.

SeqWindow_test.run2: SeqWindow_test.exe
	./SeqWindow_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 1 3 4 1000

# A wierdo one.

SeqWindow_test.run3: SeqWindow_test.exe
	./SeqWindow_test.exe $(testflags) \
		$(testdata_dir)/small.pfile 20 1 1 22

### Test some sequential windowing functions.

all_srcs += RandWindow_test.cc
all_objs += RandWindow_test.o
all_progs += RandWindow_test.exe
all_tests += RandWindow_test.run

RandWindow_test.run: RandWindow_test.run1 RandWindow_test.run2
	@true

# No margins.

RandWindow_test.run1: RandWindow_test.exe
	./RandWindow_test.exe $(testflags) \
		9 0 0 200

# One frame window with margins.

RandWindow_test.run2: RandWindow_test.exe
	./RandWindow_test.exe $(testflags) \
		1 3 4 10000

### Test the learning rate schedule ###

all_srcs += RateSchedule_test.cc
all_objs += RateSchedule_test.o
all_progs += RateSchedule_test.exe
all_tests += RateSchedule_test.run

### Test the MLP3 classes ###

all_srcs += MLP3_test.cc
all_objs += MLP3_test.o
all_progs += MLP3_test.exe
all_tests += MLP3_test.run

### Test the utilities ###

all_srcs += utils_test.cc
all_objs += utils_test.o
all_progs += utils_test.exe
all_tests += utils_test.run
garbage += rwtemp.weights

utils_test.run: utils_test.exe
	./utils_test.exe $(testflags) rwtemp.weights

### Test "mat" format output functions ###

all_srcs += mat_test.cc
all_objs += mat_test.o
all_progs += mat_test.exe
all_tests += mat_test.run
garbage += mat_test.mat

mat_test.run: mat_test.exe
	./mat_test.exe -s 10 $(testflags) mat_test.mat

### Test tanh function ###

all_srcs += tanh_test.cc
all_objs += tanh_test.o
all_progs += tanh_test.exe
all_tests += tanh_test.run
garbage += tanh_test.mat

tanh_test.run: tanh_test.exe
	./tanh_test.exe -s 10 $(testflags) tanh_test.mat


### Test convol function ###

all_srcs += convol_test.cc
all_objs += convol_test.o
all_progs += convol_test.exe
all_tests += convol_test.run
garbage += convol_test.mat

convol_test.run: convol_test.exe
	./convol_test.exe -s 10 $(testflags)


######################################################################
# The program tests
######################################################################

local_bindir = ..

check-qnsfwd: check-qnsfwd1
local_qnsfwd = $(local_bindir)/qnsfwd
local_qnstrn = $(local_bindir)/qnstrn

# Check that onlftr files and pfiles produce the same result.

check-qnsfwd1:
	$(local_qnsfwd) \
		ftr1_file=$(testdata_dir)/small.pfile \
		ftr1_format=pfile \
		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
		ftr1_ftr_start=1 \
		ftr1_ftr_count=17 \
		window_extent=9 \
		ftr1_window_offset=0 \
		ftr1_window_len=9 \
		fwd_sent_start=0 \
		fwd_sent_count=38 \
		init_weight_file=$(testdata_dir)/ntimit-rasta-phones-200.weights \
		mlp3_input_size=153 \
		mlp3_hidden_size=200 \
		mlp3_output_size=56 \
		mlp3_output_type=softmax \
		activation_format=rapbin \
		activation_file=qnsfwd1-pfile.binact && \
	$(local_qnsfwd) \
		ftr1_file=$(testdata_dir)/small.onlftr \
		ftr1_format=onlftr \
		ftr1_width=18 \
		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
		ftr1_ftr_start=1 \
		ftr1_ftr_count=17 \
		window_extent=9 \
		ftr1_window_offset=0 \
		ftr1_window_len=9 \
		fwd_sent_start=0 \
		fwd_sent_count=38 \
		init_weight_file=$(testdata_dir)/ntimit-rasta-phones-200.weights \
		mlp3_input_size=153 \
		mlp3_hidden_size=200 \
		mlp3_output_size=56 \
		mlp3_output_type=softmax \
		activation_format=rapbin \
		activation_file=qnsfwd1-onlftr.binact && \
	cmp qnsfwd1-pfile.binact qnsfwd1-onlftr.binact 

# Check that pre files and pfiles produce almost the same result.
# check-qnsfwd1a:
#	$(local_qnsfwd) \
#		ftr1_file=$(testdata_dir)/small.pfile \
#		ftr1_format=pfile \
#		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
#		ftr1_ftr_start=1 \
#		ftr1_ftr_count=17 \
#		window_extent=9 \
#		ftr1_window_offset=0 \
#		ftr1_window_len=9 \
#		fwd_sent_start=0 \
#		fwd_sent_count=38 \
#		init_weight_file=$(testdata_dir)/ntimit-rasta-phones-200.weights \
#		mlp3_input_size=153 \
#		mlp3_hidden_size=200 \
#		mlp3_output_size=56 \
#		mlp3_output_type=softmax \
#		activation_format=rapbin \
#		activation_file=qnsfwd1-pfile.binact && \
#	$(local_qnsfwd) \
#		ftr1_file=$(testdata_dir)/small.onlftr \
#		ftr1_format=onlftr \
#		ftr1_width=18 \
#		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
#		ftr1_ftr_start=1 \
#		ftr1_ftr_count=17 \
#		window_extent=9 \
#		ftr1_window_offset=0 \
#		ftr1_window_len=9 \
#		fwd_sent_start=0 \
#		fwd_sent_count=38 \
#		init_weight_file=$(testdata_dir)/ntimit-rasta-phones-200.weights \
#		mlp3_input_size=153 \
#		mlp3_hidden_size=200 \
#		mlp3_output_size=56 \
#		mlp3_output_type=softmax \
#		activation_format=rapbin \
#		activation_file=qnsfwd1-onlftr.binact && \
#	cmp qnsfwd1-pfile.binact qnsfwd1-onlftr.binact 


# Check that the "realtime" option does not change things.

check-qnsfwd2:
	$(local_qnsfwd) \
		ftr1_file=$(testdata_dir)/small.pfile \
		ftr1_format=pfile \
		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
		ftr1_ftr_start=1 \
		ftr1_ftr_count=17 \
		window_extent=9 \
		ftr1_window_offset=0 \
		ftr1_window_len=9 \
		fwd_sent_start=0 \
		fwd_sent_count=38 \
		init_weight_file=$(testdata_dir)/ntimit-rasta-phones-200.weights \
		mlp3_input_size=153 \
		mlp3_hidden_size=200 \
		mlp3_output_size=56 \
		mlp3_output_type=softmax \
		realtime=false \
		activation_format=rapbin \
		activation_file=qnsfwd2-batch.binact && \
	$(local_qnsfwd) \
		ftr1_file=$(testdata_dir)/small.pfile \
		ftr1_format=pfile \
		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
		ftr1_ftr_start=1 \
		ftr1_ftr_count=17 \
		window_extent=9 \
		ftr1_window_offset=0 \
		ftr1_window_len=9 \
		fwd_sent_start=0 \
		init_weight_file=$(testdata_dir)/ntimit-rasta-phones-200.weights \
		mlp3_input_size=153 \
		mlp3_hidden_size=200 \
		mlp3_output_size=56 \
		mlp3_output_type=softmax \
		realtime=true \
		realtime_latency=16 \
		activation_format=rapbin \
		activation_file=qnsfwd2-realtime.binact && \
	cmp qnsfwd2-batch.binact qnsfwd2-realtime.binact

# Train a net with pre file format input

check-qnstrn1: Makefile
	$(local_qnstrn) \
		ftr1_file=$(testdata_dir)/train1-rasta8+d.pre \
		ftr1_format=pre \
		ftr1_width=18 \
		ftr1_norm_file= \
		ftr1_ftr_start=1 \
		ftr1_ftr_count=17 \
		window_extent=9 \
		ftr1_window_offset=0 \
		ftr1_window_len=9 \
		hardtarget_file=$(testdata_dir)/train1-rasta8+d.pre \
		hardtarget_format=pre \
		hardtarget_window_offset=4 \
		train_sent_start=0 \
		train_sent_count=1720 \
		cv_sent_start=1720 \
		cv_sent_count=230 \
		train_cache_frames=10000 \
		train_cache_seed=0 \
		init_weight_file= \
		out_weight_file=qnstrn1.weights \
		mlp3_input_size=153 \
		mlp3_hidden_size=200 \
		mlp3_output_size=56 \
		mlp3_output_type=softmax \
		mlp3_fx=true \
		debug=0

# Soft target training

check-qnstrn2: Makefile
	$(local_qnstrn) \
		ftr1_file=$(testdata_dir)/train1-rasta8+d.pfile \
		ftr1_format=pfile \
		ftr1_width=18 \
		ftr1_norm_file=$(testdata_dir)/train1-rasta8+d.norms \
		ftr1_ftr_start=1 \
		ftr1_ftr_count=17 \
		window_extent=9 \
		ftr1_window_offset=0 \
		ftr1_window_len=9 \
		softtarget_file=$(testdata_dir)/train1-rasta8+d-targets.lna \
		softtarget_format=lna \
		softtarget_width=56 \
		softtarget_window_offset=4 \
		train_sent_start=0 \
		train_sent_count=1720 \
		cv_sent_start=1720 \
		cv_sent_count=230 \
		train_cache_frames=10000 \
		train_cache_seed=0 \
		init_weight_file= \
		out_weight_file=qnstrn1.weights \
		mlp3_input_size=153 \
		mlp3_hidden_size=200 \
		mlp3_output_size=56 \
		mlp3_output_type=softmax \
		mlp3_fx=true \
		debug=0


################################################################
# Misc targets
################################################################

$(all_objs) : ../libquicknet3.a

garbage += $(all_objs) $(all_libs) $(all_progs)

check: $(all_tests)

all: $(all_libs) $(all_progs)

clean:
	rm -f $(garbage)

distclean: clean
	rm -f Makefile

TAGS:	$(all_srcs) $(all_hdrs) Makefile
	cd $(srcdir) && etags --output=$(srcdir)/TAGS $(all_srcs) $(all_hdrs)

all_docs = TODO

################################################################
# Configuration stuff - see autoconf.info(Automatic Remaking) for details
################################################################

config_srcs = Makefile.in
config_outfiles = 

Makefile: $(srcdir)/Makefile.in ../config.status
	cp -p Makefile Makefile.old
	( cd ..; ./config.status )
	@if cmp -s Makefile Makefile.old 2>/dev/null; then \
		rm -f Makefile.old; \
		true; \
	else \
		rm -f Makefile.old; \
		echo "### Makefile changed - rerun make ###"; \
		exit 1; \
	fi
     
../config.status: $(top_srcdir)/configure
	( cd ..; ./config.status --recheck )


################################################################
# Package distribution.
################################################################


dist_files = $(all_srcs) $(all_hdrs) $(all_docs) \
	$(config_srcs) $(config_outfiles) # ChangeLog

# The name of the distribution
distname = quicknet-testsuite-$(VERSION)

# The file name of the compressed tar file
distfile = $(distname).tar.gz

# The directory in which the distribution is built
distdir = $$r/$(distname)

# Build a distribution
dist: $(distfile)

# Build the file that contains the distribution
$(distfile): disttree
	rm -f $(distfile); \
	tar cvf - $(distname) | gzip >$(distfile)

# Build a directory tree that can be tared and gzipped to form the distribution
disttree: Makefile $(dist_files)
	r=`pwd`; export r; \
	rm -fr $(distdir); \
	mkdir $(distdir); \
	chmod a=rwx $(distdir); \
	for i in $(dist_files); do \
		file=`basename $$i`; \
		cp -p $(srcdir)/$$file $(distdir)/$$file || exit 1; \
	done

