#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "hmm/hmm-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Print (phone, HMM state) pair corresponding to each frame.\n"
        "Usage:  show-transitionStates <phones-symbol-table> <transition/model-file> <alignments-rspecifier>\n"
        "e.g.: \n"
        " show-transitionStates phones.txt 1.mdl ali.1\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string phones_symtab_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3);

    //Get phones' symbol table
    fst::SymbolTable *syms = fst::SymbolTable::ReadText(phones_symtab_filename);
    if (!syms)
      KALDI_ERR << "Could not read symbol table from file "
                 << phones_symtab_filename;
    std::vector<std::string> names(syms->NumSymbols());
    for (size_t i = 0; i < syms->NumSymbols(); i++)
      names[i] = syms->Find(i);

    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    fst::SymbolTable *phones_symtab = NULL;
    {
      std::ifstream is(phones_symtab_filename.c_str());
      phones_symtab = fst::SymbolTable::ReadText(is, phones_symtab_filename);
      if (!phones_symtab || phones_symtab->NumSymbols() == 0)
        KALDI_ERR << "Error opening symbol table file "<<phones_symtab_filename;
    }
    
    
    std::ofstream alignedUtter_f;
    std::ofstream label_f;
    
    alignedUtter_f.open("labelled_utterances.txt");
    label_f.open("frameLabels.txt");

    //Start reading alignments' file
    SequentialInt32VectorReader reader(alignments_rspecifier);
    //int32 utter = 0;
    for (; !reader.Done(); reader.Next()) {

      //utter++;

      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);

      // split_str is the alignment corresponding to phone i
      std::vector<std::string> split_str(split.size());
      std::vector<std::string> split_str_phones(split.size());
      size_t i;
      //int32 sum = 0;
      for (i = 0; i < split.size(); i++) {
        std::ostringstream ss;
        std::string phone_s;
	size_t j;
        for (j = 0; j < split[i].size(); j++){
          //print Transition-Id
	  //ss << split[i][j] << " ";
	  //print Phone
	  int32 phone = trans_model.TransitionIdToPhone(split[i][j]);
	  //phone_s = phones_symtab->Find(phone);
	  ss << phone << "-"; 
	  //print HMM state
	  int32 hmm_state = trans_model.TransitionIdToHmmState(split[i][j]);	
	  ss << hmm_state << "\n";
        } 
        split_str[i] = ss.str();

	//sum = sum + j;
	
      }
      //std::alignedUtter_f<<"utterance_id: "<< " ";
      alignedUtter_f << key << endl;
      
      //uncomment to get frame number per utterance
      //alignedUtter_f << key << " ";
      //alignedUtter_f << sum << endl;
      //sum = 0;

      for (size_t i = 0; i < split_str.size(); i++)
      label_f << split_str[i];
      
      //label_f << '\n';
      
    }
    
    //alignedUtter_f << utter <<endl;

    label_f.close();
    alignedUtter_f.close();
    delete syms;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

