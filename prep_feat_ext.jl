#julia txt2julia.jl <features_filename> <#frames file> <input directory> <output directory>

ARGS = ["mfcc_to_use.txt", "cnt_fram.txt",  "/home/yannis/Desktop/new_feats--KALDI_setup/mono2","/home/yannis/Desktop/new_feats--KALDI_setup/mono2"]

using HDF5,JLD

mfcc_mat = readdlm(ARGS[3]*"/"*ARGS[1])
mfcc_mat = mfcc_mat'
m,n = size(mfcc_mat)
display((m,n))

#l = open("/home/yannis/Desktop/KALDI_norm_var/frameLabelsPDFID.txt.bin")
#labels = mmap_array(Float64, (n,1), l)

#for monophone, if no .bin available
#labels = int(readdlm("/home/yannis/Desktop/new_feats--KALDI_setup/monophone/labels.txt"))

frames_num = int(readdlm(ARGS[3]*"/"*ARGS[2]))

#choose utterances at random
#utt_sampled = int(zeros(200))
#upper = 100
#p = 1
#for i = 1:100:1000  #up to #utterances to select from
#    display(i)
    #display(upper)
#    utt_sampled_idx = vec(rand(i:upper,1,20))
#    utt_sampled[p:p-1+20] = utt_sampled_idx     #holds indices of the utterances
#    p = p + 20
#    upper = upper + 100
#    if upper > length(frames_num)
#       upper = length(frames_num)
#    end
#end

utt_sampled = [1:1500]
total_samples = sum(frames_num[1:1500])
#h5write("/home/yannis/Desktop/new_feats--KALDI_setup/monophone/utt_sampled_mono_test.h5","utt_sampled",utt_sampled)

#construct super-vectors  of features = 9 frames of MFCCs
s = open(ARGS[4]*"/"*ARGS[1]*"train.bin", "w+")
super_feat = mmap_array(Float64, (9*m,total_samples), s)
#labels_feat = zeros(total_samples)
#save dimensions of array for later opening
save(ARGS[4]*"/"ARGS[1]*"dims.jld","sv_dims",(9*m,total_samples))

k = 1
for i=1:length(utt_sampled)
   frames_num_idx = utt_sampled[i]
   starting_sample = sum(frames_num[1:frames_num_idx-1])
   super_feat[:,k] = [mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+2]' mfcc_mat[:,starting_sample+3]' mfcc_mat[:,starting_sample+4]' mfcc_mat[:,starting_sample+5]']'
   #labels_feat[k] = labels[starting_sample+1]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+2]' mfcc_mat[:,starting_sample+3]' mfcc_mat[:,starting_sample+4]' mfcc_mat[:,starting_sample+5]' mfcc_mat[:,starting_sample+6]']'
   #labels_feat[k] = labels[starting_sample+2]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+2]' mfcc_mat[:,starting_sample+3]' mfcc_mat[:,starting_sample+4]' mfcc_mat[:,starting_sample+5]' mfcc_mat[:,starting_sample+6]' mfcc_mat[:,starting_sample+7]']'
   #labels_feat[k] = labels[starting_sample+3]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+1]' mfcc_mat[:,starting_sample+2]' mfcc_mat[:,starting_sample+3]' mfcc_mat[:,starting_sample+4]' mfcc_mat[:,starting_sample+5]' mfcc_mat[:,starting_sample+6]' mfcc_mat[:,starting_sample+7]' mfcc_mat[:,starting_sample+8]']'
   #labels_feat[k] = labels[starting_sample+4]
   k = k + 1
   for j = starting_sample+5 : starting_sample+frames_num[frames_num_idx]-5
      super_feat[:,k] = [mfcc_mat[:,j-4]' mfcc_mat[:,j-3]' mfcc_mat[:,j-2]' mfcc_mat[:,j-1]' mfcc_mat[:,j]' mfcc_mat[:,j+1]' mfcc_mat[:,j+2]' mfcc_mat[:,j+3]' mfcc_mat[:,j+4]']'
      #labels_feat[k] = labels[j]
      k = k + 1
   end
   super_feat[:,k] = [mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-8]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-7]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-6]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-5]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-4]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-3]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-2]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-1]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]']'
   #labels_feat[k] = labels[starting_sample+frames_num[frames_num_idx]-4]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-7]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-6]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-5]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-4]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-3]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-2]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-1]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]']'
   #labels_feat[k] = labels[starting_sample+frames_num[frames_num_idx]-3]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-6]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-5]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-4]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-3]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-2]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-1]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]']'
   #labels_feat[k] = labels[starting_sample+frames_num[frames_num_idx]-2]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-5]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-4]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-3]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-2]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-1]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]']'
   #labels_feat[k] = labels[starting_sample+frames_num[frames_num_idx]-1]
   k = k + 1
   super_feat[:,k] = [mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-4]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-3]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-2]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]-1]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]' mfcc_mat[:,starting_sample+frames_num[frames_num_idx]]']'
   #labels_feat[k] = labels[starting_sample+frames_num[frames_num_idx]]
   k = k + 1
end

write(s,super_feat)
msync(super_feat)
close(s)

h5write("/home/yannis/Desktop/new_feats--KALDI_setup/mono2/mfcc_train117.h5","mfcc_sampled",super_feat)
#h5write("/home/yannis/Desktop/new_feats--KALDI_setup/monophone/labels_sampled_mono_test.h5","labels_sampled",labels_feat)
