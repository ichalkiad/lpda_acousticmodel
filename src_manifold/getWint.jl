using MAT,MATLAB,HDF5,JLD,Clustering,KDTrees

#Parameters for matrix construction: number of neighbors and distance type 
kint = 100   #30
Rint = 850
distance = "euclidean"

#Read in 13-d MFCC dataset,corresponding labels, and result of tabulate() on labels
triphones_labels = load("/media/data/ichalkia/knn13/phones_priors_PhonesVis.mat")["phones_priorsPhones5k"]
mfcc_sampled = h5read("/media/data/ichalkia/knn13/PhoneVisData13.h5","data")'
labels_sampled = h5read("/media/data/ichalkia/knn13/PhoneVisLabels.h5","labels")

#triphone clustering for Wint
distinct_labels = triphones_labels[:,1]
feat_indices_l = int(zeros(size(distinct_labels,1)))
for i = 1 : size(distinct_labels,1)
  feat_indices = find(labels_sampled .== distinct_labels[i])
  feat_indices_l[i] = length(feat_indices)
  #if kint>sizeof(feat_indices)
  #  display(i)
  #end
  #Comment out following line, if run more than once on same dataset 
  h5write("/media/data/ichalkia/knn13/labels_idx_pdfid_phonesStatesVis/$i.h5","idx",feat_indices)
end
display(minimum(feat_indices_l))

data = mfcc_sampled

p = size(data,2)
Wint = speye(p,p)
Wint_dense = zeros(kint)
nn_idx_dense = zeros(kint)


for i = 1 : size(distinct_labels,1)
    display(i)
	idx = h5read("/media/data/ichalkia/knn13/labels_idx_pdfid_phonesStatesVis/$i.h5", "idx")
    if (length(idx)==0)
	continue
    else
        tree = KDTree(data[:,idx])
        r = 10.0
        for j = 1 : length(idx)
        	current_feat = data[:,idx[j]]
        	current_label = labels_sampled[idx[j]]
		#Uncomment for exact neighbors
        	#(nn_idx,nn_dist) = knn(tree, current_feat, kint+kpen)
	        nn_idx = inball(tree, vec(current_feat), r, false)

        	search_times=0        
        	while (length(nn_idx) < kint+1)
              		r = 2*r
              		#display(j)
		        #(nn_idx,nn_dist) = knn(tree, current_feat, kint+kpen)
              		nn_idx = inball(tree, current_feat, r, false)
	      		search_times = search_times + 1
	      		if (search_times==5)
				break
				display("Class with few members")
	      		end
        	end
        	r = 10.0

	      	nn_idx_dense = idx[nn_idx]
        	nn_idx_dense_noself = find(nn_idx_dense .!= idx[j])
                #Keep number of found same-class neighbours, it is <= kint
		kint = min(kint,length(nn_idx_dense_noself))

		nn_idx_dense = nn_idx_dense[nn_idx_dense_noself][1:kint]

       		d = zeros(kint)
        	for m = 1 : kint
            		if distance=="euclidean"
            			d[m] = norm(data[:,nn_idx_dense[m]]-current_feat,2).^2
		        elseif distance=="correlation"
            			d[m] = 1 - dot(vec(data[:,nn_idx_dense[m]]),vec(current_feat))/(norm(data[:,nn_idx_dense[m]],2)*norm(current_feat,2))
          		end
	        end
        	Wint_dense = exp(-d./Rint)
        	Wint_sp = sparsevec(vec(nn_idx_dense),vec(Wint_dense),p)
        	Wint[:,idx[j]] = Wint_sp
	end
    end
end

h5write("/media/data/ichalkia/knn13/WintLPDA30_Vis.h5","Wint",full(Wint))
#save("/media/data/ichalkia/knn13/WintLPDA30_phonesStates.jld", "Wint", Wint)

