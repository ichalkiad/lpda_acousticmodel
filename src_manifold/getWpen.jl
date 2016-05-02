using MAT,MATLAB,HDF5,JLD,Clustering,KDTrees

#Parameters for matrix construction: number of neighbors and distance type 
kpen = 70 #30
Rpen = 3000
distance = "euclidean"

Read in 13-d MFCC dataset and corresponding labels
mfcc_sampled = h5read("/media/data/ichalkia/knn13/PhoneVisData13.h5","data")'
labels_sampled = h5read("/media/data/ichalkia/knn13/PhoneVisLabels.h5","labels")


#kmeans clustering for Wpen
#Set number of clusters according to dataset size, min_cluster_num == 2
k = 2 #25
data = mfcc_sampled 

p = size(data,2)
Wpen = speye(p,p)
Wpen_dense = zeros(kpen)
nn_idx_dense_pen = zeros(kpen)

seeds  = initseeds(:rand,data,k)
result = kmeans(data,k,init=seeds)
display(counts(result))

##get neighbours and embed
cluster_idx = assignments(result)

for i = 1 : k
    display(i)
    idx_same_cluster = vec(find(cluster_idx .== i)) ##same idx as in data
    tree = KDTree(data[:,idx_same_cluster])
    r = 25.0

    for j = 1 : length(idx_same_cluster)

        current_feat = vec(data[:,idx_same_cluster[j]])
        current_label = labels_sampled[idx_same_cluster[j]]
	#Uncomment for exact neighbors
        #(nn_idx,nn_dist) = knn(tree, current_feat, kpen)
        nn_idx = inball(tree, current_feat, r, false)
        nn_labels = labels_sampled[idx_same_cluster[nn_idx]]
        nn_diff_labels = find(nn_labels .!= current_label)
	nn_idx_dense_pen = idx_same_cluster[nn_idx][nn_diff_labels][1:kpen]
        d = zeros(kpen)
        for m = 1 : kpen
            if distance=="euclidean"
              d[m] = norm(vec(data[:,nn_idx_dense_pen[m]])-current_feat,2).^2
            elseif distance=="correlation"
              d[m] = 1 - dot(vec(data[:,nn_idx_dense_pen[m]]),vec(current_feat))/(norm(data[:,nn_idx_dense_pen[m]],2)*norm(current_feat,2))
            end
        end
	
	Wpen_dense = exp(-d./Rpen)
	
        Wpen_sp = sparsevec(vec(nn_idx_dense_pen),vec(Wpen_dense),p)
        Wpen[:,idx_same_cluster[j]] = Wpen_sp
    end
end

h5write("/media/data/ichalkia/knn13/WpenLPDA30_Vis.h5","Wpen",full(Wpen))
#save("/media/data/ichalkia/knn13/WpenLPDA30_phonesStates.jld", "Wpen", Wpen)
