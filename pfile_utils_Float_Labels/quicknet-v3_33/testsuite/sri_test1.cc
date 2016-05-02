// $Header: /u/drspeech/repos/quicknet2/testsuite/sri_test1.cc,v 1.4 2013/11/01 00:33:53 davidj Exp $
//
// Read tests for SRI list files.
// davidj - 5th March 1996

#include <assert.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

#include "QuickNet.h"
#include "rtst.h"

void
sri_read_test(int debug, const char* srifile, const char* dir, size_t num_sents, size_t num_ftrs)
{
    size_t i;			// Counter
    int ec;			// Error code

    rtst_start("sri_test1");
    FILE* testfile = QN_open(srifile, "r");

    ec = chdir(dir);
    assert(ec == 0);

    QN_InFtrStream_ListSRI sf(debug, "srifile", testfile, 1);

    // Check the basic attributes of the PFile
    rtst_assert(sf.num_segs()==num_sents);
    rtst_assert(sf.num_ftrs()==num_ftrs);



    // Read through the whole PFile, checking lots of details Here we assume
    // the PFile holds normal speech data - that labels hold small positive
    // integers and features are small reals.  We also sum all the number of
    // frames in each sentence, and ensure the total is the sum as the number
    // of frames in the whole database
    {
	enum { FRAMES = 17 }; // Choose an obscure number of frames to read
	float* ftrs;
	size_t num_ftr_vals = FRAMES * num_ftrs;
	const float bad_ftr = 1e30;
	const float min_ftr = -30.0f;
	const float max_ftr = 30.0f;
	QN_SegID sentid;
	size_t acc_frames = 0;

	ftrs = rtst_padvec_new_vf(num_ftr_vals);
	for (i=0; i<num_sents; i++)
	{
	    size_t sent_frames;
	    size_t frame;
	    size_t cnt = FRAMES;
	    
	    sentid = sf.nextseg();
	    rtst_assert(sentid!=QN_SEGID_BAD);
	    sent_frames = sf.num_frames(i);
	    assert(sent_frames!=QN_SIZET_BAD);
	    acc_frames += sent_frames;

	    for (frame=0; frame<sent_frames; frame+=cnt)
	    {

		// Clear the ftr and lab vectors before reading
		qn_copy_f_vf(num_ftr_vals, bad_ftr, ftrs);

		cnt = sf.read_ftrs(FRAMES, ftrs);
		rtst_assert(cnt!=QN_SIZET_BAD);

		// Check that everything we read is sensible
		rtst_checkrange_ffvf(cnt*num_ftrs, min_ftr, max_ftr, ftrs);

		// Check that stuff we did not change is not corrupted
		if (cnt!=FRAMES)
		{
		    rtst_checkrange_ffvf((FRAMES-cnt)*num_ftrs,
					 bad_ftr, bad_ftr,
					 &ftrs[num_ftrs*cnt]);
		}
	    }
	}
	sentid = sf.nextseg();
	rtst_assert(sentid==QN_SEGID_BAD);
	rtst_padvec_del_vf(ftrs);
    }
    // Check seeking to start of PFile
    {
	int ec;
	size_t current_sent, current_frame;
	QN_SegID id;

	ec = sf.rewind();
	rtst_assert(ec==QN_OK);
	// Test that get_pos after rewind() works.
	ec = sf.get_pos(&current_sent, &current_frame);
	rtst_assert(ec==QN_OK);
	rtst_assert(current_sent==QN_SIZET_BAD);
	rtst_assert(current_frame==QN_SIZET_BAD);
	// Test that get_pos with NULL pointers works.
	ec = sf.get_pos(NULL, NULL);

	id = sf.nextseg();
	rtst_assert(id!=QN_SEGID_BAD);
	ec = sf.get_pos(&current_sent, &current_frame);
	rtst_assert(ec==QN_OK);
	rtst_assert(current_sent==0);
	rtst_assert(current_frame==0);
    }
    // Some storage for feature vectors
    float* last_frame_seek = new float [num_ftrs];
    float* last_frame_read = new float [num_ftrs];

    // Seek to end of PFile and return last frame
    {
	int ec;
	size_t current_sent, current_frame;

	// Goto end of PFile and check returned frame
	size_t last_sent  = num_sents - 1;
	size_t last_frame = sf.num_frames(last_sent) - 1;
	ec = sf.set_pos(last_sent, last_frame);
	rtst_assert(ec!=QN_SEGID_BAD);
	ec = sf.get_pos(&current_sent, &current_frame);
	rtst_assert(ec==QN_OK);
	rtst_assert(current_sent==last_sent);
	rtst_assert(current_frame==last_frame);

	size_t cnt = sf.read_ftrs(1, last_frame_seek);
	rtst_assert(cnt==1);
    }
    // Scan through PFile and read last frame
    {
	size_t current_sent, current_frame;
	int id;

	id = sf.set_pos(0, 0);
	rtst_assert(id!=QN_SEGID_BAD);
	id = sf.get_pos(&current_sent, &current_frame);
	rtst_assert(id!=QN_SEGID_BAD);
	rtst_assert(current_sent==0);
	rtst_assert(current_frame==0);
	for (i=0; i<num_sents-1; i++)
	{
	    QN_SegID sentid;
	    sentid = sf.nextseg();
	    rtst_assert(sentid!=QN_SEGID_BAD);
	}
	for (i=0; i<sf.num_frames(num_sents-1)-1; i++)
	{
	    size_t count;
	    count = sf.read_ftrs(1, NULL);
	    rtst_assert(count==1);
	}
	size_t cnt = sf.read_ftrs(1, last_frame_read);
	rtst_assert(cnt==1);
	rtst_assert(sf.read_ftrs(1,NULL)==0);
	rtst_assert(sf.nextseg()==QN_SEGID_BAD);
    }
    rtst_checkeq_vfvf(num_ftrs, last_frame_seek, last_frame_read);
    delete [] last_frame_seek;
    delete [] last_frame_read;


    QN_close(testfile);

    rtst_passed();
}


int
main(int argc, char* argv[])
{
    int arg;
    int debug = 0;		// Verbosity level

    arg = rtst_args(argc, argv);
    QN_logger = new QN_Logger_Simple(rtst_logfile, stderr,
				     "sri_test1");

    assert(arg==argc-4);
    const char* srifile = argv[arg++];
    const char* dir = argv[arg++];
    long num_sents = strtol(argv[arg++], NULL, 0);
    long num_ftrs = strtol(argv[arg++], NULL, 0);
    if (rtst_logfile!=NULL)
	debug = 99;
    sri_read_test(debug, srifile, dir, num_sents, num_ftrs); 
    rtst_exit();
}
