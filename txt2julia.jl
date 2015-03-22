#julia <filename> <input directory> <output directory>

using HDF5,JLD

#ARGS[1] = "/home/yannis/Desktop/LPDA/raw_mfcc_train_si284_1_txt_formatted.ark"

#default delimiter is whitespace
feature_mat = readdlm(ARGS[2]*"/"*ARGS[1])'
#save(ARGS[1]*".jld","features",feature_mat)

m,n = size(feature_mat)

#construct super-vectors  of features = 9 frames of MFCCs
s = open(ARGS[3]*"/"*ARGS[1]*".bin", "w+")
super_feat = mmap_array(Float64, (9*m,n), s)

#for first/last 5 super_vec? Same vectors but different labels??
#super_feat[1,:] = [mfccs[1,:] mfccs[2,:] mfccs[3,:] mfccs[4,:] mfccs[5,:] mfccs[6,:] mfccs[7,:] mfccs[8,:] mfccs[9,:]]

for i = 5:m-4
    super_feat[:,i] = [feature_mat[:,i-4]' feature_mat[:,i-3]' feature_mat[:,i-2]' feature_mat[:,i-1]' feature_mat[:,i]' feature_mat[:,i+1]' feature_mat[:,i+2]' feature_mat[:,i+3]' feature_mat[:,i+4]']
end

msync(super_feat)
close(s)

#Needed????
#save("super_vectors.jld","super_feats",super_feat);
