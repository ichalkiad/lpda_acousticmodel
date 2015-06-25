#julia txt2julia_labels.jl <labels_filename> <input directory> <output directory>

using HDF5,JLD


(m1,n1) = load("/home/yannis/Desktop/KALDI_norm_var/mfcc_normMV_nodelim.txtdims.jld")["sv_dims"]

#construct labels vector
s = open(ARGS[3]*"/"*ARGS[1]*".bin", "w+")
labels = mmap_array(Float64, (n1,1), s)
labels = readdlm(ARGS[2]*"/"*ARGS[1])
write(s,labels)
close(s)
