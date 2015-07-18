#KALDI feature normalization :  ../../../src/featbin/apply-cmvn --norm_vars --norm_means --utt2spk=ark:data/train_si284/utt2spk scp:mfcc/cmvn_train_si284.scp scp:data/train_si284/feats.scp ark:mfcc_norm.ark

using MAT,MATLAB,HDF5,JLD,Clustering,KDTrees

(m,n) = load("/home/yannis/Desktop/KALDI_norm_var/mfcc_normMV_nodelim.txtdims.jld")["sv_dims"]
#s = open("/home/yannis/Desktop/KALDI_norm_var/mfcc_normMV_nodelim.txt.bin")

#mfcc = mmap_array(Float64, (m,n), s)
#l = open("/home/yannis/Desktop/KALDI_norm_var/frameLabels2a.txt.bin")
#labels = mmap_array(Float64, (n,1), l)

kint = 14
kpen = 14
Rint = 1000
Rpen = 3000
distance = "correlation"

##Sample
#mfcc_sampled = zeros(117,900000)
#labels_sampled = zeros(900000)
#upper = 1000000
#m = 1
#for i = 1:1000000:30000000
#    mfcc_sampled_idx = vec(rand(i:upper,1,30000))
#    mfcc_sampled[:,m:m-1+30000] = mfcc[:,mfcc_sampled_idx]
#    labels_sampled[m:m-1+30000] = labels[mfcc_sampled_idx]
#    m = m + 30000
#    upper = upper + 1000000
#   if upper > n
#       upper = n
#    end
#end
#h5write("/home/yannis/Desktop/KALDI_norm_var/mfcc_sampled1M_norm_mvK.h5", "sampled_mfcc", mfcc_sampled)
#h5write("/home/yannis/Desktop/KALDI_norm_var/labels_sampled1M_norm_mvK.h5", "sampled_labels", labels_sampled)
mfcc_sampled = h5read("/home/yannis/Desktop/KALDI_norm_var/mfcc_sampled1M_norm_mvK.h5", "sampled_mfcc")
labels_sampled = h5read("/home/yannis/Desktop/KALDI_norm_var/labels_sampled1M_norm_mvK.h5", "sampled_labels")
triphones_labels = load("/home/yannis/Desktop/KALDI_norm_var/phones_priors.mat")["phones_priors"]

##cluster sample dataset

#triphone clustering for Wint
#watch out for size of distinct_labels[i], might not have enough neighbours in the current sampling, kint/kpen < min(feat_indices_l)
distinct_labels = triphones_labels[:,1]
feat_indices_l = int(zeros(size(distinct_labels,1)))
for i = 1 : size(distinct_labels,1)
  feat_indices = find(labels_sampled .== distinct_labels[i])
  feat_indices_l[i] = length(feat_indices)
  #if kint>sizeof(feat_indices)
  #  display(i)
  #end
  #h5write("/home/yannis/Desktop/labels_idx/$i.h5", "idx", feat_indices)
end
minimum(feat_indices_l)

#kmeans clustering for Wpen
k = 13
data = mfcc_sampled

p = size(data,2)
Wint = speye(p,p)
Wint_dense = zeros(kint)
nn_idx_dense = zeros(kint)
Wpen = speye(p,p)
Wpen_dense = zeros(kpen)
nn_idx_dense_pen = zeros(kpen)

seeds = initseeds(:rand,data,k)
result = kmeans(data,k,init=seeds)
display(result.counts)



##get neighbours and embed
cluster_idx = assignments(result)

for i = 1 : k
    display(i)
    idx_same_cluster = find(cluster_idx .== i) ##same idx as in data
    tree = KDTree(data[:,idx_same_cluster])
    r = 25.0

    for j = 1 : length(idx_same_cluster)

        current_feat = data[:,idx_same_cluster[j]]
        current_label = labels_sampled[idx_same_cluster[j]]
        #(nn_idx,nn_dist) = knn(tree, current_feat, kint+kpen)
        nn_idx = inball(tree, current_feat, r, false)
        nn_labels = labels_sampled[idx_same_cluster[nn_idx]]
        nn_diff_labels = find(nn_labels .!= current_label)
        nn_idx_dense_pen = idx_same_cluster[nn_idx][nn_diff_labels][1:kpen]

        d = zeros(kpen)
        for m = 1 : kpen
            if distance=="euclidean"
              d[m] = norm(data[:,nn_idx_dense_pen[m]]-current_feat,2).^2
            elseif distance=="correlation"
              d[m] = 1 - dot(data[:,nn_idx_dense_pen[m]],current_feat)
            end
        end
        Wpen_dense = exp(-d./Rpen)

        Wpen_sp = sparsevec(vec(nn_idx_dense_pen),vec(Wpen_dense),p)
        Wpen[:,idx_same_cluster[j]] = Wpen_sp
    end
end


for i = 1 : size(distinct_labels,1)
    display(i)
    idx = h5read("/home/yannis/Desktop/labels_idx/$i.h5", "idx")
    tree2 = KDTree(data[:,idx])
    r = 10.0
    for j = 1 : length(idx)

        current_feat = data[:,idx[j]]
        current_label = labels_sampled[idx[j]]

        #(nn_idx,nn_dist) = knn(tree, current_feat, kint+kpen)
        nn_idx = inball(tree2, current_feat, r, false)
        #nn_labels = labels_sampled[idx[nn_idx]]
        #nn_same_labels = find(nn_labels .== current_label) #redundant here, just check

        while (length(nn_idx) < kint+1)
              r = 2*r
              display(j)
              #(nn_idx,nn_dist) = knn(tree, current_feat, kint+kpen)
              nn_idx = inball(tree2, current_feat, r, false)
              #nn_labels = labels_sampled[idx[nn_idx]]
              #nn_same_labels = find(nn_labels .== current_label)
        end
        r = 10.0

        nn_idx_dense = idx[nn_idx]#[nn_same_labels]
        nn_idx_dense_noself = find(nn_idx_dense .!= idx[j])
        nn_idx_dense = nn_idx_dense[nn_idx_dense_noself][1:kint]

        d = zeros(kint)
        for m = 1 : kint
            if distance=="euclidean"
              d[m] = norm(data[:,nn_idx_dense[m]]-current_feat,2).^2
            elseif distance=="correlation"
              d[m] = 1 - dot(data[:,nn_idx_dense[m]],current_feat)/(norm(data[:,nn_idx_dense[m]],2)*norm(current_feat,2))
            end
        end
        Wint_dense = exp(-d./Rint)

        Wint_sp = sparsevec(vec(nn_idx_dense),vec(Wint_dense),p)
        Wint[:,idx[j]] = Wint_sp

    end

end




