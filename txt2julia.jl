using HDF5,JLD

function txt2hdf5(mfcc_dir)

    #mfcc_dir = "/home/yannis/Desktop/LPDA";
    feature_mat = readdlm("$mfcc_dir/feat.txt")';
    save("$mfcc_dir/feat.jld","features",feature_mat);
    return feature_mat;

end
