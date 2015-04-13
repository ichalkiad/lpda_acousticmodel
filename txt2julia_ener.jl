#julia txt2julia.jl <features_filename> <input directory> <output directory>

using HDF5,JLD

#default delimiter is whitespace
feature_mat = readdlm(ARGS[2]*"/"*ARGS[1])'
m,n = size(feature_mat)

#construct energy features = 9 frames of MFCC energies
s = open(ARGS[3]*"/"*ARGS[1]*"_ener"*".bin", "w+")
super_feat = mmap_array(Float64, (9,n), s)

#save dimensions of array for later opening
save(ARGS[3]*"/"ARGS[1]*"_ener"*"dims.jld","sv_dims",(9,n))

for i = 1:m
    if i<5
      super_feat[:,i] = [feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i+1]' feature_mat[1,i+2]' feature_mat[1,i+3]' feature_mat[1,i+4]']'
      write(s,super_feat[:,i])
    elseif i>m-4
      super_feat[:,i] = [feature_mat[1,i-4]' feature_mat[1,i-3]' feature_mat[1,i-2]' feature_mat[1,i-1]' feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i]' feature_mat[1,i]']'
      write(s,super_feat[:,i])
    else
      super_feat[:,i] = [feature_mat[1,i-4]' feature_mat[1,i-3]' feature_mat[1,i-2]' feature_mat[1,i-1]' feature_mat[1,i]' feature_mat[1,i+1]' feature_mat[1,i+2]' feature_mat[1,i+3]' feature_mat[1,i+4]']'
      write(s,super_feat[:,i])
    end
end

write(s,super_feat)
msync(super_feat)
close(s)
