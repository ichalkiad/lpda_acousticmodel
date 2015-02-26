using HDF5,JLD

mfcc_file = ARGS[1];

function txt2hdf5(mfcc_file)

   #default delimiter is whitespace
   feature_mat = readdlm("$mfcc_file")';
   save("$mfcc_file.jld","features",feature_mat);
   return feature_mat;

end

mfccs = txt2hdf5(mfcc_file);
