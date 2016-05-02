# $Header: /u/drspeech/repos/quicknet2/Makefile.in,v 1.173 2013/10/29 04:16:42 davidj Exp $
#
# Makefile.  Generated from Makefile.in by configure.
#
# Makefile.in for quicknet - a 3 layer MLP library for SPERT
# davidj - Mon Jun  5 11:18:46 1995
#


# Standard things we need

.SUFFIXES: 
.SUFFIXES: .c .cc .o .lo .s .dis .i .cu

SHELL=/bin/sh

# Details of our host and target

build_alias = 
host_alias = 

srcdir = .
top_srcdir = .
builddir = .
top_builddir = .

subdirs = testsuite perfsuite

# Installation stuff

version = v3_33
prefix = /media/data/ichalkia/kaldi-trunk/tools/quicknet-v3_33
exec_prefix = ${prefix}
libdir = $(exec_prefix)/lib
bindir = $(exec_prefix)/bin
includedir = $(exec_prefix)/include
qnincludedir = $(includedir)/quicknet3
mandir = $(prefix)/man
man1dir = $(mandir)/man1
man3dir = $(mandir)/man3
man5dir = $(mandir)/man5
docdir = $(prefix)/doc
pcdir = $(libdir)/pkgconfig
qndocdir = $(docdir)/quicknet-$(version)
transform = s,x,x,

# Programs we use

SHELL = /bin/bash
CVS = cvs
CC = gcc
CXX = g++
NVCC = 
LD = ${CXX}
RANLIB = ranlib
LIBTOOL = $(SHELL) $(top_builddir)/libtool
INSTALL = /usr/bin/install -c
INSTALL_PROG = $(INSTALL)
INSTALL_DATA = ${INSTALL} -m 644
INSTALL_LIB = $(INSTALL_DATA)

# Tool arguments

CPPFLAGS = 
CFLAGS = -g -O2
CXXFLAGS = -g -O2
NVCCFLAGS = 
LDFLAGS = 
LIBS = -lpthread -lm 
INCS = 

cflags = -I$(builddir) -I$(srcdir) $(CFLAGS) $(CPPFLAGS)
cxxflags =  -I$(builddir) -I$(srcdir) $(CXXFLAGS) $(CPPFLAGS)
nvccflags =  -I$(builddir) -I$(srcdir) $(NVCCFLAGS) $(CPPFLAGS)
ldflags = -L. $(LDFLAGS)
libs =  $(LIBS)
compile.c = $(CC) $(cflags)
compile.cc = $(CXX) $(cxxflags)
compile.cu = $(NVCC) $(nvccflags)
shcompile.c = $(LIBTOOL) --mode=compile $(CC) $(cflags)
shcompile.cc = $(LIBTOOL) --mode=compile $(CXX) $(cxxflags)
shar = $(LIBTOOL) --mode=link $(LD)

################################################################
# Implicit rules
################################################################

.c.o:
	$(compile.c) -c $<

.c.lo:
	$(shcompile.c) -c $<

.c.i:
	$(compile.c) -E $< > $@

.cc.o:
	$(compile.cc) -c $<

.cc.lo:
	$(shcompile.cc) -c $<

.cc.i:
	$(compile.cc) -E $< > $@

.cu.o:
	$(compile.cu) -c $<


################################################################
# The default target
################################################################

default: all


################################################################
# Variables we accumulate into
################################################################

all_srcs =
all_objs =
all_shobjs =
all_hdrs =
all_stlibs =
all_shlibs =
all_progs =
all_man1srcs =
all_man1pages =
all_man3srcs =
all_man3pages =
all_man5srcs =
all_man5pages =
all_docs =
all_configured_scripts =
all_installed_scripts =
garbage =
distgarbage =
rcs_files =

######################################################################
# The library itself
######################################################################
qnlib_c_srcs = \
	$(srcdir)/QN_libc.c
qnlib_cxx_srcs = \
	$(srcdir)/QN_misc.cc \
	$(srcdir)/QN_Logger.cc \
	$(srcdir)/QN_Logger_Simple.cc \
	$(srcdir)/QN_defaultlogger.cc \
	$(srcdir)/QN_PFile.cc \
	$(srcdir)/QN_intvec.cc \
	$(srcdir)/QN_fltvec.cc \
	$(srcdir)/QN_fltvec_convol.cc \
	$(srcdir)/QN_fltvec_bmul3.cc \


qnlib_cu_srcs = \
	$(srcdir)/QN_cuvec.cu \
	$(srcdir)/QN_CudaUtils.cu \


qnlib_hdrs = \
	$(srcdir)/QN_types.h \
	$(srcdir)/QN_args.h \
	$(srcdir)/QN_args2.h \
	$(srcdir)/QN_libc.h \
	$(srcdir)/QuickNet.h \
	$(srcdir)/QN_Logger.h \
	$(srcdir)/QN_Logger_Simple.h \
	$(srcdir)/QN_utils.h \
	$(srcdir)/QN_PFile.h \
	$(srcdir)/QN_streams.h \
	$(srcdir)/QN_intvec.h \
	$(srcdir)/QN_fltvec.h \
	$(srcdir)/QN_cuvec.h \
	$(srcdir)/QN_CudaUtils.h

qnlib_core_objs = \
	QN_libc.o \
	QN_args.o \
	QN_args2.o \
	QN_misc.o \
	QN_Logger.o \
	QN_Logger_Simple.o \
	QN_defaultlogger.o \
	QN_PFile.o \
        QN_intvec.o \
	QN_fltvec.o \
	QN_fltvec_convol.o \
	QN_fltvec_omul.o \
	QN_fltvec_bmul1.o \
	QN_fltvec_bmul2.o \
	QN_fltvec_bmul3.o \
	

qnlib_cuda_objs = \
	QN_cuvec.o \
	QN_CudaUtils.o \
	

qnlib_opt_objs = 
qnlib_objs = $(qnlib_core_objs) $(qnlib_opt_objs)

qnlib_shobjs = \
	QN_libc.lo \
	QN_args.lo \
	QN_args2.lo \
	QN_misc.lo \
	QN_Logger.lo \
	QN_Logger_Simple.lo \
	QN_defaultlogger.lo \
	QN_PFile.lo \
	QN_intvec.lo \
	QN_fltvec.lo \
	QN_fltvec_convol.lo \
	QN_fltvec_omul.lo \
	QN_fltvec_bmul1.lo \
	QN_fltvec_bmul2.lo \
	QN_fltvec_bmul3.lo \


$(qnlib_objs): $(qnlib_hdrs)  QN_config.h

$(qnlib_shobjs): $(qnlib_hdrs)  QN_config.h

#QN_ftrcut.o: $(srcdir)/QN_cut.cc
#	$(compile.cc) -c -DFTRCUT $(srcdir)/QN_cut.cc -o QN_ftrcut.o

#QN_labcut.o: $(srcdir)/QN_cut.cc
#	$(compile.cc) -c -DLABCUT $(srcdir)/QN_cut.cc -o QN_labcut.o

#QN_ftrwindows.o: $(srcdir)/QN_windows.cc
#	$(compile.cc) -c -DFTRWIN $(srcdir)/QN_windows.cc -o QN_ftrwindows.o

#QN_labwindows.o: $(srcdir)/QN_windows.cc
#	$(compile.cc) -c -DLABWIN $(srcdir)/QN_windows.cc -o QN_labwindows.o

#QN_ftrcut.lo: $(srcdir)/QN_cut.cc
#	$(shcompile.cc) -c -DFTRCUT $(srcdir)/QN_cut.cc -o QN_ftrcut.o

#QN_labcut.lo: $(srcdir)/QN_cut.cc
#	$(shcompile.cc) -c -DLABCUT $(srcdir)/QN_cut.cc -o QN_labcut.o

#QN_ftrwindows.lo: $(srcdir)/QN_windows.cc
#	$(shcompile.cc) -c -DFTRWIN $(srcdir)/QN_windows.cc -o QN_ftrwindows.o

#QN_labwindows.lo: $(srcdir)/QN_windows.cc
#	$(shcompile.cc) -c -DLABWIN $(srcdir)/QN_windows.cc -o QN_labwindows.o

qnlib_name = quicknet3
qnlib_file = lib$(qnlib_name).a
qnshlib_file = lib$(qnlib_name).la

$(qnlib_file) : $(qnlib_objs)
	$(shar) -o $@ $(qnlib_objs)

$(qnshlib_file) : $(qnlib_shobjs)
	$(shar) -o $@ $(qnlib_shobjs) -rpath $(libdir)

QN_config.h: config.h $(srcdir)/QN_config_tail.h
	rm -f QN_config.h
	sed -e 's/#define /#define QN_/' config.h > QN_config.h \
		|| rm -f QN_config.h
	cat $(srcdir)/QN_config_tail.h >>QN_config.h \
		|| rm -f QN_config.h

distgarbage += QN_config.h

all_srcs += $(qnlib_c_srcs) $(qnlib_cxx_srcs) $(qnlib_cu_srcs)
all_objs += $(qnlib_objs)
all_shobjs += $(qnlib_shobjs)
all_hdrs += $(qnlib_hdrs)
all_stlibs += $(qnlib_file)
all_shlibs += $(qnshlib_file)
install_hdrs += $(qnlib_hdrs) QN_config.h

################################################################
# A demonstration of how to use the library
################################################################

qnxor.o : $(srcdir)/qnxor.cc $(qnlib_hdrs) \
					QN_config.h
	$(compile.cc) -c $(srcdir)/qnxor.cc

qnxor : qnxor.o $(qnlib_file)
	$(LD) $(ldflags) -o qnxor qnxor.o $(qnlib_file) \
		 $(libs)

all_srcs += $(srcdir)/qnxor.cc
all_objs += qnxor.o
# all_progs += qnxor

################################################################
# A test of the QN_Range object
################################################################

################################################################
# A program for calculating norms of a feature file
################################################################
# A program for adding deltas to a feature file (using QN_fir)
################################################################

################################################################
# A program for converting between wts file formats (using QN_MLWe*)
################################################################


################################################################
# The new, full-featured training program
################################################################


################################################################
# The new, full-featured forward pass program
################################################################



################################################################
# Other man pages
################################################################

################################################################
# A training program for 2->5 layer nets
################################################################


################################################################
# A forward pass program for 2->5 layer nets
################################################################



################################################################
# A hacked test program
################################################################
##############################################################
# A script to select the best version of qnstrn on the current arch
################################################################

qndo.sh: stamp-h

all_configured_scripts += qndo.sh.in

all_installed_scripts += qndo

distgarbage += qndo.sh

################################################################
# Some scripts for running qnstrn etc. on default test data
################################################################
# Actual executable scripts are built by running config.status

testdata_qnnorm.sh: stamp-h

testdata_qnstrn.sh: stamp-h

testdata_qnsfwd.sh: stamp-h

testdata_qnstrn_ps.sh: stamp-h

testdata_qnsfwd_ps.sh: stamp-h

testdata_qnmultitrn.sh: stamp-h

testdata_qnmultifwd.sh: stamp-h

all_configured_scripts += \
	$(srcdir)/testdata_qnnorm.sh.in \
	$(srcdir)/testdata_qnstrn.sh.in \
	$(srcdir)/testdata_qnsfwd.sh.in \
	$(srcdir)/testdata_qnstrn_ps.sh.in \
	$(srcdir)/testdata_qnsfwd_ps.sh.in \
	$(srcdir)/testdata_qnmultitrn.sh.in \
	$(srcdir)/testdata_qnmultifwd.sh.in

distgarbage += \
	testdata_qnnorm.sh \
	testdata_qnstrn.sh \
	testdata_qnsfwd.sh \
	testdata_qnstrn_ps.sh \
	testdata_qnsfwd_ps.sh \
	testdata_qnmultitrn.sh \
	testdata_qnmultifwd.sh

################################################################
# Configuration stuff - see autoconf.info(Automatic Remaking) for details
################################################################


$(srcdir)/configure: $(srcdir)/configure.in $(srcdir)/aclocal.m4
	cd ${srcdir} && autoconf

# autoheader might not change config.h.in, so touch a stamp file.
$(srcdir)/config.h.in: $(srcdir)/stamp-h.in

$(srcdir)/stamp-h.in: $(srcdir)/configure.in $(srcdir)/aclocal.m4
	( cd $(srcdir) && autoheader )
	echo timestamp > $(srcdir)/stamp-h.in

config.status: $(srcdir)/configure
	./config.status --recheck

config.h: stamp-h

Makefile: stamp-h

QN3Config.sh: stamp-h

quicknet3.pc: stamp-h

quicknet3-install.pc: stamp-h

libtool: stamp-h

stamp-h: config.status $(srcdir)/config.h.in $(srcdir)/Makefile.in \
				QN3Config.sh.in quicknet3.pc.in \
				quicknet3-install.pc.in \
				$(all_configured_scripts) 
	cp -p Makefile Makefile.old
	./config.status
	@if cmp -s Makefile Makefile.old 2>/dev/null; then \
		rm -f Makefile.old; \
		true; \
	else \
		rm -f Makefile.old; \
		echo "### Makefile changed - rerun make ###"; \
		exit 1; \
	fi

distgarbage += stamp-h config.status config.cache libtool Makefile \
	QN3Config.sh QN3Config-install.sh quicknet3.pc quicknet3-install.pc \
	config.h config.log
config_srcs = \
	$(srcdir)/configure.in \
	$(srcdir)/aclocal.m4 \
	$(srcdir)/Makefile.in \
	$(srcdir)/QN3Config.sh.in \
	$(srcdir)/quicknet3.pc.in \
	$(srcdir)/quicknet3-install.pc.in \
	$(srcdir)/QN_config_tail.h \
	$(srcdir)/install-sh \
	$(srcdir)/config.guess \
	$(srcdir)/config.sub \
	$(srcdir)/mkinstalldirs \
	$(srcdir)/ltmain.sh
config_outfiles = \
	$(srcdir)/configure \
	$(srcdir)/config.h.in \
	$(srcdir)/stamp-h.in

QN3Config-install.sh: QN3Config.sh
	sed -e 's/{QN_BUILD_/{QN_INSTALL_/' QN3Config.sh \
		>QN3Config-install.sh || rm -f QN3Config-install.sh

all_docs += $(srcdir)/INSTALL $(srcdir)/README $(srcdir)/NEWS \
	$(srcdir)/TODO $(srcdir)/COPYING $(srcdir)/README-hacker \
	$(srcdir)/INSTALL-icsi \
	$(srcdir)/newargs.txt $(srcdir)/perf.txt $(srcdir)/sphinx.txt \
	$(srcdir)/hardware.txt \
	$(all_man1srcs) $(all_man3srcs) $(all_man5srcs) iconfigure.sh


################################################################
# Misc targets
################################################################

garbage += $(all_objs) $(all_shobjs) $(all_stlibs) $(all_shlibs) $(all_progs)
garbage += *.o *.lo a.out core
garbage += TAGS

# Let configure decide which libraries to use based on --enable-shared
all_libs =  $(all_stlibs)

all: $(all_libs) $(all_progs) quicknet3.pc quicknet3-install.pc \
	QN3Config-install.sh QN3Config.sh

libs: $(all_libs)

clean:
	@if [ "no" = "yes" -a -d testsuite ]; then \
		cd testsuite; \
		$(MAKE) clean; \
	fi
	rm -f $(garbage)

distclean: 
	@if [ "no" = "yes" -a -d testsuite ]; then \
		cd testsuite; \
		$(MAKE) clean; \
	fi
	rm -f $(garbage) $(distgarbage)

check: $(all_libs)
	@if [ "no" = "yes" -a -d testsuite ]; then \
		cd testsuite && $(MAKE) check; \
	else \
		echo "### Sorry - no rtst library - cannot run QuickNet library testsuite ###"; \
	fi

install: install-progs install-libs install-docs

install-libs: Makefile  $(all_libs) $(install_hdrs) QN3Config-install.sh \
						quicknet3-install.pc
	$(srcdir)/mkinstalldirs $(libdir) $(pcdir) \
			$(includedir) $(qnincludedir)
	for i in $(all_libs); do \
		case $$i in \
		*.a) \
			$(INSTALL_LIB) $$i $(libdir)/$$i; \
			$(RANLIB) $(libdir)/$$i \
			;; \
		*.la) \
			$(LIBTOOL) --mode=install $(INSTALL) \
				$$i $(libdir)/$$i \
			;; \
		esac; \
	done
	for i in $(install_hdrs); do \
		$(INSTALL_DATA) $$i $(qnincludedir)/`basename $$i`; \
	done
	$(INSTALL_PROG) QN3Config-install.sh $(libdir)/QN3Config.sh
	$(INSTALL_DATA) quicknet3-install.pc $(pcdir)/quicknet3.pc

install-progs: Makefile  $(all_progs) $(all_man1srcs) $(all_man3srcs) $(all_man5srcs)
	$(srcdir)/mkinstalldirs $(bindir) $(mandir) $(man1dir) $(man3dir) $(man5dir)
	for i in $(all_progs); do \
		$(INSTALL_PROG) $$i $(bindir)/`echo $$i|sed '$(transform)'`; \
	done
	for i in $(all_installed_scripts); do \
		$(INSTALL_PROG) $$i.sh $(bindir)/`echo $$i|sed '$(transform)'`; \
	done
	for i in $(all_man1pages); do \
		$(INSTALL_DATA) $(srcdir)/$$i.man \
			$(man1dir)/`echo $$i|sed '$(transform)'`.1; \
	done
	for i in $(all_man3pages); do \
		$(INSTALL_DATA) $(srcdir)/$$i.man \
			$(man3dir)/`echo $$i|sed '$(transform)'`.3; \
	done
	for i in $(all_man5pages); do \
		$(INSTALL_DATA) $(srcdir)/$$i.man \
			$(man5dir)/`echo $$i|sed '$(transform)'`.5; \
	done

install-docs: Makefile  $(srcdir)/NEWS
	$(srcdir)/mkinstalldirs $(docdir) $(qndocdir)
	$(INSTALL_DATA) $(srcdir)/NEWS $(qndocdir)/NEWS.txt
	$(INSTALL_DATA) $(srcdir)/TODO $(qndocdir)/TODO.txt
	$(INSTALL_DATA) $(srcdir)/README $(qndocdir)/README.txt
	$(INSTALL_DATA) $(srcdir)/README-hacker $(qndocdir)/README-hacker.txt
	$(INSTALL_DATA) $(srcdir)/INSTALL $(qndocdir)/INSTALL.txt

TAGS:	$(all_srcs) $(all_hdrs) Makefile
	etags -C --output=$(srcdir)/TAGS $(all_srcs) $(all_hdrs)


################################################################
# Version control etc.
################################################################

rcs_files += $(all_srcs) $(all_hdrs) $(all_docs) $(config_srcs) \
	$(all_configured_scripts)



################################################################
# Package distribution.
################################################################

dist_files = $(all_srcs) $(all_hdrs) $(all_docs) \
	$(config_srcs) $(config_outfiles) \
	$(all_configured_scripts) # $(srcdir)/ChangeLog

# The name of the distribution
name = quicknet
distname = $(name)-$(version)

# The file name of the compressed tar file
distfile = $(distname).tar.gz

# The directory in which the distribution is built
distdir = $(top_builddir)/$(distname)

# Build a distribution
dist: $(distfile)

# Build the file that contains the distribution
$(distfile): disttree
	rm -f $(distfile); \
	tar cvf - $(distname) | gzip >$(distfile)

# Build a directory tree that can be tared and gzipped to form the distribution
disttree: Makefile $(dist_files)
	rm -fr $(distdir); \
	mkdir $(distdir); \
	chmod a=rwx $(distdir); \
	for i in $(dist_files); do \
		file=`basename $$i`; \
		cp -p $(srcdir)/$$file $(distdir)/$$file || exit 1; \
	done; \
	distdir=`cd $(distdir); pwd`; \
	for i in $(subdirs); do \
	( \
		cd $$i; \
		$(MAKE) distdir=$${distdir}/$$i disttree \
	) \
	done


