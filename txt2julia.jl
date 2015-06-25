#julia txt2julia.jl <features_filename> <input directory> <output directory>

#ARGS = ["mfcc_si284_nodelim.txt","/home/yannis/Desktop/GraphEmbedding","/home/yannis/Desktop"]

using HDF5,JLD

#default delimiter is whitespace
feature_mat = readdlm(ARGS[2]*"/"*ARGS[1])
#save(ARGS[3]*"/"*ARGS[1]*".jld","features",feature_mat)

#custom normalization!
#m = mean(feature_mat,1)
#sigma = std(feature_mat,1)
#feature_mat = (feature_mat .- m)./sigma
#save(ARGS[3]*"/"*ARGS[1]*"_norm.jld","features",feature_mat)

###feature_mat = load("/home/yannis/Desktop/feature_mat_norm.jld","features")

feature_mat = feature_mat'
m,n = size(feature_mat)

#construct super-vectors  of features = 9 frames of MFCCs
s = open(ARGS[3]*"/"*ARGS[1]*".bin", "w+")
super_feat = mmap_array(Float64, (9*m,n), s)
#save dimensions of array for later opening
save(ARGS[3]*"/"ARGS[1]*"dims.jld","sv_dims",(9*m,n))


for i = 1:n
    if i<5
      super_feat[:,i] = [feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i+1]' feature_mat[:,i+2]' feature_mat[:,i+3]' feature_mat[:,i+4]']'
      write(s,super_feat[:,i])
      ##OFFSET OF feature vector 1
      ##display(position(s))

    elseif i>n-4
      super_feat[:,i] = [feature_mat[:,i-4]' feature_mat[:,i-3]' feature_mat[:,i-2]' feature_mat[:,i-1]' feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i]' feature_mat[:,i]']'
      write(s,super_feat[:,i])
    else
      super_feat[:,i] = [feature_mat[:,i-4]' feature_mat[:,i-3]' feature_mat[:,i-2]' feature_mat[:,i-1]' feature_mat[:,i]' feature_mat[:,i+1]' feature_mat[:,i+2]' feature_mat[:,i+3]' feature_mat[:,i+4]']'
      write(s,super_feat[:,i])
    end
end

#write(s,super_feat)
msync(super_feat)
close(s)
